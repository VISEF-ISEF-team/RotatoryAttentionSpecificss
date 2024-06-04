import cv2
from sklearn.model_selection import train_test_split
from glob import glob
import os
import numpy as np
from torch.utils.data import Dataset


class RotatoryModelDataset(Dataset):
    def __init__(self, images_folder, labels_folder):

        self.samples = []

        images_folder_list = sorted(os.listdir(images_folder))
        labels_folder_list = sorted(os.listdir(labels_folder))

        self.n_samples = 0

        for order in images_folder_list:

            # loop through both images and labels

            for image_path, label_path in zip(os.listdir(os.path.join(images_folder, order)), os.listdir(os.path.join(labels_folder, order))):

                single_sample = {}
                single_sample["index"] = self.n_samples

                # create a single order
                single_sample["order"] = order

                # create image path
                single_sample["image_path"] = os.path.join(
                    images_folder, order, image_path)

                # create folder path
                single_sample["label_path"] = os.path.join(
                    labels_folder, order, label_path)

                self.samples.append(single_sample)

                self.n_samples += 1

    def __len__(self):
        return self.n_samples

    def read_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image / 255.0
        image = image.astype(np.float32)
        image = np.expand_dims(
            image, axis=0).astype(np.float32)

        return image

    def read_label(self, label_path):
        label = np.load(label_path)
        label = np.expand_dims(label, axis=0).astype(np.uint8)

        return label

    def __getitem__(self, index):
        # read image
        image = self.read_image(self.samples[index]["image_path"])

        # read label
        label = self.read_label(self.samples[index]["label_path"])

        return image, label


class NormalModelDataset(Dataset):
    def __init__(self, images_path, labels_path, num_classes=8):
        self.images_path = images_path
        self.labels_path = labels_path
        self.num_classes = num_classes
        self.n_samples = len(self.images_path)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)
        image = image / 255.0
        image = image.astype(np.float32)
        image = np.expand_dims(
            image, axis=0).astype(np.float32)

        label = np.load(self.labels_path[index])
        label = np.expand_dims(label, axis=0).astype(np.uint8)

        return image, label


def main():
    pass


if __name__ == "__main__":
    main()
