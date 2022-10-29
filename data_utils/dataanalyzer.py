import os

import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetParser(object):
    '''
    Class to parse the data_utils and split it into train and test sets.
    '''

    def __init__(self, labeltxt_p, images_dir):
        """
        Initializes the data_utils parser.
        :param images_dir: The path to the images.
        """
        self.labels_p = labeltxt_p
        self.images_dir = images_dir
        self.characters = set()
        self.max_len = 10
        self.df_dataset = self._load_dataset()
        self.char_to_labels = {char: idx for idx, char in enumerate(self.characters)}
        self.labels_to_char = {val: key for key, val in self.char_to_labels.items()}
        self.random_seed = 1234

    def _load_dataset(self):
        """
        Loads the dataset2 from the given path.
        :return: A list of images.
        """

        images = []
        labels = []

        with open(self.labels_p, 'r') as file:
            lines = file.readlines()
            for line in lines:
                try: 
                    line = line.strip("\n").split("\t")
                    image_base_name = os.path.basename(line[0])
                    image_path = os.path.join(self.images_dir, image_base_name)
                    image_path = os.path.abspath(image_path)
                    images.append(image_path)
                    labels.append(line[1])
                    for ch in line[1]:
                        self.characters.add(ch)
                except Exception as e:
                    print(line)
                    print(str(e))
        self.characters = sorted(list(self.characters))
        self.max_len = max(list(map(len, labels)))
        pd_dataset = pd.DataFrame({"images": images, "labels": labels}, index=None)
        return pd_dataset.sample(frac=1.).reset_index(drop=True)

    def split_train_validation(self):
        training_data, validation_data = train_test_split(self.df_dataset, test_size=0.2, random_state=self.random_seed)
        return training_data, validation_data


def main():
    DATASET_PATH = '/dataset/gt.txt'
    a = DatasetParser(DATASET_PATH).char_to_labels
    print(a)


if __name__ == "__main__":
    main()
