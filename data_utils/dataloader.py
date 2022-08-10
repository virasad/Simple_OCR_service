import numpy as np
import torch.utils.data as Data
from PIL import Image


class OCRDataloader(Data.Dataset):
    def __init__(self, images, labels, transform):
        """
        Args:
            data_path: path to the data
            label_path: path to the label it would be a txt file in the format of:
                        image_path label
                        image_path label
                        ...
            preprocess: a function to preprocess the image
            transform: transform to be applied on a sample
        """
        print("Loading data...")
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        # print(self.images)
        img = Image.open(self.images[index])
        label = self.labels[index]
        # normalize the image
        img = np.array(img)
        # img = img / 255.0
        img = self.transform(image=np.asarray(img))
        return img["image"], label

    def __len__(self):
        return len(self.images)

class PredictOCRDataloader(Data.Dataset):
    def __init__(self, images, transform):
        """
        Args:
            images: list of image paths or list of PIL images

            preprocess: a function to preprocess the image
            transform: transform to be applied on a sample
        """
        print("Loading data...")
        self.images = images
        self.transform = transform
        if not isinstance(self.images, list):
            self.images = [self.images]
        self.image_type = type(images[0])
        if self.image_type == str:
            self.imread = Image.open

        elif self.image_type == np.ndarray:
            self.imread = lambda x: x

    def __getitem__(self, index):
        # print(self.images)
        img = self.imread(self.images[index])
        # normalize the image
        img = np.array(img)
        # img = img / 255.0
        img = self.transform(image=np.asarray(img))
        return img["image"]

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    pass