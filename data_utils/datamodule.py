import os

import albumentations as A
import pytorch_lightning as pl
# TO tensor
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from .dataanalyzer import DatasetParser
from .dataloader import OCRDataloader, PredictOCRDataloader


class OCRDataModule(pl.LightningDataModule):
    def __init__(self, labels_txt=None, images='./', img_width: int = 32, img_height: int = 32, batch_size: int = 32,
                 mode='train'):
        super().__init__()
        """
        Args:
            labels_txt: path to the labels txt file in the predict mode do not require this argument
            images: path of the images dir in the predict mode it should be path of the images or list of np.ndarray  
            img_width: width of the image
            img_height: height of the image
            batch_size: batch size of the data loader
            mode: train or predict
        """
        self.labels_file = labels_txt
        # check for valid path

        self.images = images
        # check for valid path
        if mode != 'predict':
            if not os.path.isfile(self.labels_file):
                raise ValueError("Invalid path to labels file")
            if not os.path.isdir(self.images):
                raise ValueError("Invalid path to images directory")
        # load dataset
        # augmentations
        self.batch_size = batch_size
        self.train_augmentations = A.Compose([
            A.PadIfNeeded(min_height=img_height, min_width=img_width, always_apply=True),
            A.RandomCrop(height=img_height, width=img_width, always_apply=True),
            A.Normalize(always_apply=True),
            # A.GaussNoise(p=0.1),
            # A.Blur(p=0.1),
            # A.JpegCompression(p=0.1),
            # A.RandomRain(p=0.1),
            # A.RandomBrightness(p=0.1),
            # A.RandomContrast(p=0.1),
            # A.RandomGamma(p=0.1),
            # A.ToGray(always_apply=True),
            ToTensorV2(always_apply=True),
        ])
        self.val_augmentations = A.Compose([
            A.PadIfNeeded(min_height=img_height, min_width=img_width, always_apply=True),
            A.RandomCrop(height=img_height, width=img_width, always_apply=True),
            A.Normalize(always_apply=True),
            # A.ToGray(always_apply=True),
            ToTensorV2(always_apply=True),
        ])
        if mode != 'predict':
            self.dataset_parser = DatasetParser(self.labels_file, self.images)
            self.train_dataset, self.val_dataset = self.dataset_parser.split_train_validation()
            self.characters = self.dataset_parser.characters
            self.num_classes = len(self.characters)
            self.max_len = self.dataset_parser.max_len

    def setup(self, stage: str = 'fit'):
        if stage == 'fit':
            self.train_dataset_len = len(self.train_dataset)
            self.val_dataset_len = len(self.val_dataset)
            self.train_loader = OCRDataloader(images=self.train_dataset["images"].array,
                                              labels=self.train_dataset["labels"].array,
                                              transform=self.train_augmentations)

            self.val_loader = OCRDataloader(images=self.val_dataset["images"].array,
                                            labels=self.val_dataset["labels"].array,
                                            transform=self.val_augmentations)

        elif stage == 'test':
            self.test_dataset = self.dataset_parser.split_train_validation()[1]
            self.test_loader = OCRDataloader(images=self.test_dataset["images"].array,
                                             labels=self.test_dataset["labels"].array,
                                             transform=self.val_augmentations)

        elif stage == 'predict':
            self.predict_loader = PredictOCRDataloader(images=self.images,
                                                           transform=self.val_augmentations)

        else:
            raise ValueError("Invalid stage")

    def train_dataloader(self):
        return DataLoader(self.train_loader, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_loader, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_loader, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_loader, batch_size=self.batch_size, shuffle=False)
