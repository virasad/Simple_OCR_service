# import trainer
import pytorch_lightning as pl

from data_utils import datamodule
from model.model import LitOCR
from data_utils.custom_cb import ClientLogger

# load datamodule


def ocr_trainer(img_w, img_h, labels_txt_p, images_path, save_dir, model_name, max_epochs=10,
                lr=1e-3,
                batch_size=128,
                log_url=None,
                task_id=None):
    hidden_size = 256

    data_module = datamodule.OCRDataModule(labels_txt=labels_txt_p,
                                           images=images_path,
                                           img_width=img_w,
                                           img_height=img_h,
                                           batch_size=batch_size)
    data_module.setup(stage='fit')
    model = LitOCR(img_w=img_w,
                   img_h=img_h,
                   characters=data_module.characters,
                   input_channel=3,
                   hidden_size=hidden_size,
                   output_channel=len(data_module.characters),
                   batch_max_length=data_module.max_len,
                   optimizer_name='Adam',
                   lr=lr,
                   decode='greedy')

    custom_cb = ClientLogger(log_url, task_id, max_epochs=max_epochs)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        filename=model_name + '_{epoch}_{val_loss:.2f}_{val_acc:.2f}', )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=100,
        verbose=True,
        mode='min')

    stochastic_weight_averaging = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=max_epochs,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback, stochastic_weight_averaging, early_stop_callback],
        logger=custom_cb,
    )

    trainer.fit(model, data_module)


def test():
    ocr_trainer(img_w=128, img_h=64, labels_txt_p='dataset/gt.txt', images_path='dataset/test',
                save_dir='checkpoints/', model_name='model2', max_epochs=10, lr=1e-3, batch_size=256
                )


if __name__ == '__main__':
    test()
