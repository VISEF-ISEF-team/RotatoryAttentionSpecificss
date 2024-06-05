import cv2
import os
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset

# TODO: implement one hot encode on the fly and remove it out of the fucking training loop


class RotatoryModelDataset(Dataset):
    def __init__(self, images_folder: list, labels_folder: list, num_classes: int):
        self.num_classes = num_classes
        self.samples = []

        self.n_samples = 0

        for root_image_folder, root_label_folder in zip(images_folder, labels_folder):

            order = os.path.basename(root_image_folder)

            # loop through the files in the folder
            for image_path, label_path in zip(sorted(os.listdir(root_image_folder)), sorted(os.listdir(root_label_folder))):

                single_sample = {}
                single_sample["index"] = self.n_samples

                # create a single order
                single_sample["order"] = order

                # create image path
                single_sample["image_path"] = os.path.join(
                    root_image_folder, image_path)

                # create folder path
                single_sample["label_path"] = os.path.join(
                    root_label_folder, label_path)

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
        label = torch.from_numpy(label)

        label = F.one_hot(label.long(), num_classes=self.num_classes)
        label = torch.squeeze(label, dim=1)
        label = label.permute(2, 0, 1)

        return label

    def __getitem__(self, index):
        # read image
        image = self.read_image(self.samples[index]["image_path"])

        # read label
        label = self.read_label(self.samples[index]["label_path"])

        return image, label


class NormalModelDataset(Dataset):
    def __init__(self, images_path: list, labels_path: list, num_classes: int):
        self.num_classes = num_classes

        self.images_path = images_path
        self.labels_path = labels_path
        self.n_samples = len(self.images_path)

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
        label = torch.from_numpy(label)

        label = F.one_hot(label.long(), num_classes=self.num_classes)
        label = torch.squeeze(label, dim=1)
        label = label.permute(2, 0, 1)

        return label

    def __getitem__(self, index):
        # get image
        image = self.read_image(self.images_path[index])

        # get label
        label = self.read_label(self.labels_path[index])

        return image, label


def main():
    pass


if __name__ == "__main__":
    main()
