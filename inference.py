import pytorch_lightning as pl

from data_utils.datamodule import OCRDataModule
from model.model import LitOCR


class InferenceOCR():
    def __init__(self):
        self.batch_size = 1

    def set_model(self, model_path):
        self.model = LitOCR.load_from_checkpoint(checkpoint_path=model_path)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def inference(self, images):
        self.data_module = OCRDataModule(images=images,
                                         img_width=self.model.hparams.img_w,
                                         img_height=self.model.hparams.img_h,
                                         batch_size=self.batch_size,
                                         mode='predict')

        res = pl.trainer.Trainer(gpus=1).predict(self.model, datamodule=self.data_module)
        res_dict = {
                    'preds': [],
                    'conf': []
                    }

        for result in res:
            res_dict['preds'].extend(result['preds'])
            res_dict['conf'].extend(result['conf'])
        return res_dict


def test(model_path, images):
    inference = InferenceOCR()
    inference.set_model(model_path)
    res = inference.inference(images)
    return res


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    checkpoint_path = 'checkpoints/model_epoch=156_val_loss=0.27_val_acc=0.74.ckpt'
    images = ['dataset/test/image-401.png', 'dataset/test/image-402.png', 'dataset/test/image-403.png']
    np_images = []
    for image in images:
        np_image = np.array(Image.open(image))
        np_images.append(np_image)
    res = test(checkpoint_path, np_images)
    print(res)