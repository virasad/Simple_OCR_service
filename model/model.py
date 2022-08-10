import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance

from data_utils.encode_decode import CTCLabelConverter


class VGG_FeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # 256x8x25
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))  # 512x1x24

    def forward(self, input):
        return self.conv_net(input)


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        try:
            self.rnn.flatten_parameters()
        except:
            pass
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class CRNN(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_classes):
        super(CRNN, self).__init__()
        self.feature_extractor = VGG_FeatureExtractor(input_channel, output_channel)
        self.feature_extractor_output = output_channel
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sequence_model = nn.Sequential(
            BidirectionalLSTM(input_size=self.feature_extractor_output,
                              hidden_size=hidden_size,
                              output_size=hidden_size),
            BidirectionalLSTM(input_size=hidden_size,
                              hidden_size=hidden_size,
                              output_size=hidden_size)
        )

        self.sequence_model_output = hidden_size
        self.prediction_layer = nn.Linear(self.sequence_model_output, num_classes)

    def forward(self, input):
        visual_feature = self.feature_extractor(input)
        visual_feature = self.adaptive_avg_pool(
            visual_feature.permute(0, 3, 1, 2))  # batch_size x output_channel x h x w
        visual_feature = visual_feature.squeeze(3)  # batch_size x output_channel x h
        contextual_feature = self.sequence_model(visual_feature)
        prediction = self.prediction_layer(contextual_feature)
        return prediction


class LitOCR(pl.LightningModule):
    def __init__(self, img_w, img_h, characters, input_channel, hidden_size, lr ,output_channel=512,

                 batch_max_length=34,
                 optimizer_name="Adam",
                 decode='greedy', ):
        """
        {"lr": 1, "rho": 0.95,"eps": 1e-7}
        Args:
            :param feature_extractor: a string indicating the feature extractor to use
        """
        super().__init__()
        self.save_hyperparameters()

        # check the w and h of the image
        assert img_w % 16 == 0 and img_h % 16 == 0, 'imgH and imgW has to be a multiple of 16'
        # self.model = model
        self.img_w = img_w
        self.img_h = img_h
        self.criterion = torch.nn.CTCLoss(zero_infinity=True)
        self.converter = CTCLabelConverter(characters)
        self.num_class = len(self.converter.character)
        self.best_accuracy = -1
        self.best_norm_ed = -1
        self.model = CRNN(input_channel, output_channel, hidden_size, self.num_class)
        self.device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError("optimizer_name not implemented")
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        # print(labels)
        batch_size = images.size(0)
        text, length = self.converter.encode(labels, batch_max_length=self.hparams.batch_max_length)
        preds = self.model(images)
        # print(images[0], 'preds \n' )
        # print(images[1])
        preds = preds.log_softmax(2)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.permute(1, 0, 2)
        torch.backends.cudnn.enabled = False
        loss = self.criterion(preds, text, preds_size, length)
        # print("loss: ", preds)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        n_correct = 0
        norm_ED = 0
        images, labels = val_batch
        batch_size = images.size(0)
        length_for_pred = torch.IntTensor([self.hparams.batch_max_length] * batch_size)
        text_for_pred = torch.LongTensor(batch_size, self.hparams.batch_max_length + 1).fill_(0)
        text_for_loss, length_for_loss = self.converter.encode(labels, batch_max_length=self.hparams.batch_max_length)
        preds = self.model(images)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        loss = self.criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)
        self.log('val_loss', loss)
        if self.hparams.decode == 'greedy':
            print('greedy')
            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = self.converter.decode_greedy(preds_index.data, preds_size.data)
            print(preds_str, '\n' , labels)
        elif self.hparams.decode == 'beamsearch':
            preds_str = self.converter.decode_beamsearch(preds, beamWidth=2)
            print('beamsearch', preds_str, '\n', labels)
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if pred == gt:
                n_correct += 1
            # print(gt, pred_max_prob)

            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])

        self.log('val_acc', n_correct / batch_size, prog_bar=True)
        self.log('val_norm_ED', norm_ED / batch_size, prog_bar=True)
        self.log('val_confidence_score', confidence_score, prog_bar=True)
        return {'val_loss': loss, 'val_acc': n_correct / batch_size, 'val_norm_ED': norm_ED / batch_size,
                'val_confidence_score': confidence_score, 'val_pred': preds_str}

    def predict_step(self, pred_batch, batch_idx):
        images = pred_batch
        batch_size = images.size(0)
        preds = self.model(images)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        if self.hparams.decode == 'greedy':
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = self.converter.decode_greedy(preds_index.data, preds_size.data)
        elif self.hparams.decode == 'beamsearch':
            preds_str = self.converter.decode_beamsearch(preds, beamWidth=2)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidences = []
        for pred_max_prob in preds_max_prob:
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidences.append(float(confidence_score.cpu().detach().numpy()))

        return {"preds": preds_str, "conf": confidences}







