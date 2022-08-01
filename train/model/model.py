import pytorch_lightning as pl
import torch
import torchvision.models as models
from torch import nn
from torch.nn import functional as F


def get_model(model_name, pretrained=True):
    # remove the final fc layer
    if model_name == 'vgg':
        model = models.vgg16(pretrained=pretrained)
        del model.classifier[6]
        return model

    elif model_name == 'resnet':
        model = models.resnet18(pretrained=pretrained)
        del model.fc
        return model

    elif model_name == 'densenet':
        model = models.densenet161(pretrained=pretrained)
        del model.classifier[1]
        return model

    elif model_name == 'resnext':
        model = models.resnext50_32x4d(pretrained=pretrained)
        del model.fc
        return model

    elif model_name == 'wideresnet':
        model = models.wide_resnet50_2(pretrained=pretrained)
        del model.fc
        return model

    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=pretrained)
        del model.classifier
        model.conv_last = nn.Conv2d(1280, 1, kernel_size=1, stride=1, padding=0)
        return model
    elif model_name == 'mnasnet':
        model = models.mnasnet1_0(pretrained=pretrained)
        del model.classifier[1]
        return model


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class LitOCR(pl.LightningModule):
    def __init__(self, feature_extractor, img_w, img_h):
        """
        Args:
            :param feature_extractor: a string indicating the feature extractor to use
        """
        super().__init__()
        assert feature_extractor in ['vgg', 'resnet', 'densenet', 'resnext', 'wideresnet', 'mobilenetv2',
                                     'mnasnet']  # check if the feature extractor is valid
        # check the w and h of the image
        assert img_w % 16 == 0 and img_h % 16 == 0, 'imgH and imgW has to be a multiple of 16'
        self.feature_extractor = get_model(feature_extractor)
        self.img_w = img_w
        self.img_h = img_h
        self.encoder = BidirectionalLSTM(4096, 512, 512)
        self.criterion = torch.nn.CTCLoss(zero_infinity=True)


    def forward(self, x):
        embedding = self.feature_extractor(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)


if __name__ == '__main__':
    # load the data
    a = get_model('mobilenetv2', False)
    print(a)
