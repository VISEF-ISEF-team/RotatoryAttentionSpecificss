import os
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
import nibabel as nib
from torchvision import transforms
from train_support import duplicate_end, duplicate_open_end, get_slice_from_volumetric_data
import cv2


class RotatoryModelDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(self.images_path)

    def __len__(self):
        return self.n_samples

    def convert_label_to_class(self, mask):
        lookup_table = {
            0.0: 0.0,
            500.0: 1.0,
            600.0: 2.0,
            420.0: 3.0,
            550.0: 4.0,
            205.0: 5.0,
            820.0: 6.0,
            850.0: 7.0,
        }

        for i in np.unique(mask):
            mask[mask == i] = lookup_table[i]

    def __getitem__(self, index):
        image = nib.load(self.images_path[index]).get_fdata()
        mask = nib.load(self.masks_path[index]).get_fdata()

        self.convert_label_to_class(mask)

        return image, mask


class NormalModelDataset(Dataset):
    def __init__(self, images_path, masks_path, num_classes=8):
        self.images_path = images_path
        self.masks_path = masks_path
        self.num_classes = num_classes
        self.n_samples = len(self.images_path)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image = image / 255.0
        image = image.astype(np.float32)
        image = np.expand_dims(
            image, axis=0).astype(np.float32)

        mask = np.load(self.masks_path[index])
        mask = torch.from_numpy(mask)
        mask = F.one_hot(mask, num_classes=self.num_classes)
        mask = mask.numpy()
        mask = resize(mask, (256, 256, 8),
                      preserve_range=True, anti_aliasing=True)
        mask = torch.from_numpy(mask)
        mask = torch.argmax(mask, dim=-1)
        mask = torch.unsqueeze(mask, dim=0)

        return image, mask


def load_dataset(images, masks, split=0.2):
    split_size = int(split * len(images))

    x_train, x_val = train_test_split(
        images, test_size=split_size, random_state=42)

    y_train, y_val = train_test_split(
        masks, test_size=split_size, random_state=42)

    return (x_train, y_train), (x_val, y_val)


def get_rotatory_loaders(root_images, root_labels, split=0.2, num_workers=6):
    (x_train, y_train), (x_val, y_val) = load_dataset(
        root_images, root_labels, split=split)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    train_dataset = RotatoryModelDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=None, shuffle=True, num_workers=num_workers)

    val_dataset = RotatoryModelDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=None,
                            shuffle=False, num_workers=num_workers)

    return (train_loader, val_loader)


def get_normal_loaders(root_images, root_labels, batch_size, split=0.2, num_workers=6, num_classes=8):

    (x_train, y_train), (x_val, y_val) = load_dataset(
        root_images, root_labels, split=split)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    train_dataset = NormalModelDataset(
        x_train, y_train, num_classes=num_classes)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = NormalModelDataset(x_val, y_val, num_classes=num_classes)
    val_loader = DataLoader(val_dataset, batch_size=20,
                            shuffle=False, num_workers=num_workers)

    return (train_loader, val_loader)


def get_single_batch_normal_loader(root_images, root_labels, batch_size, split=0.2, num_workers=6, num_classes=8):
    (x_train, y_train), (x_val, y_val) = load_dataset(
        root_images, root_labels, split=split)

    random_train_indices = np.random.randint(
        low=0, high=len(x_train), size=batch_size).astype(int)

    random_val_indices = np.random.randint(
        low=0, high=len(x_val), size=batch_size).astype(int)

    x_train = [x_train[i] for i in random_train_indices]
    y_train = [y_train[i] for i in random_train_indices]
    x_val = [x_val[i] for i in random_val_indices]
    y_val = [y_val[i] for i in random_val_indices]

    train_dataset = NormalModelDataset(
        x_train, y_train, num_classes=num_classes)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = NormalModelDataset(x_val, y_val, num_classes=num_classes)
    val_loader = DataLoader(val_dataset, batch_size=20,
                            shuffle=False, num_workers=num_workers)

    return (train_loader, val_loader)


def get_single_batch_rotatory_loader(root_images, root_labels, split=0.2, num_workers=6):
    (x_train, y_train), (x_val, y_val) = load_dataset(
        root_images, root_labels, split=split)

    random_train_index = np.random.randint(low=0, high=len(x_train))
    random_val_index = np.random.randint(low=0, high=len(x_val))

    x_train = [x_train[random_train_index]]
    y_train = [y_train[random_train_index]]

    x_val = [x_val[random_val_index]]
    y_val = [y_val[random_val_index]]

    train_dataset = RotatoryModelDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=None, shuffle=True, num_workers=num_workers)

    val_dataset = RotatoryModelDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=None,
                            shuffle=False, num_workers=num_workers)

    return (train_loader, val_loader)


def test_rotatory():
    root = "../Triple-View-R-Net/data_for_training/MMWHS/"

    root_images = sorted(
        glob(os.path.join(root, "ct_train", "images", "*.nii.gz")))
    root_labels = sorted(
        glob(os.path.join(root, "ct_train", "masks", "*.nii.gz")))

    train_loader, val_loader = get_rotatory_loaders(
        root_images=root_images, root_labels=root_labels)

    train_transform_trivial = transforms.Compose([
        transforms.TrivialAugmentWide(num_magnitude_bins=5),
    ])

    for x, y in train_loader:

        x = duplicate_open_end(x)
        y = duplicate_open_end(y)

        length = x.shape[-1]

        print(f"Image Volume: {x.shape} || Mask Volume: {y.shape}")

        for i in range(0, length, 7):

            if i + 8 >= length:
                num_slice = length - i
                if num_slice < 3:
                    for i in range(3 - num_slice):
                        x = duplicate_end(x)
                        y = duplicate_end(y)

                    num_slice = 3

            else:
                num_slice = 8

            images, masks = get_slice_from_volumetric_data(
                x, y, i, num_slice)

            print(images.shape, masks.shape)

        print("-" * 30)


def test_normal():
    root = "../Triple-View-R-Net/data_for_training/MMWHS/"

    root_images = sorted(
        glob(os.path.join(root, "images", "*.png")))
    root_labels = sorted(
        glob(os.path.join(root, "masks", "*.npy")))

    train_loader, val_loader = get_normal_loaders(
        root_images=root_images, root_labels=root_labels, batch_size=8)

    for x, y in train_loader:
        print(f"Image: {x.shape} || Mask: {y.shape}")


def test_single_batch_normal():
    root = "../Triple-View-R-Net/data_for_training/MMWHS/"

    root_images = sorted(
        glob(os.path.join(root, "images", "*.png")))
    root_labels = sorted(
        glob(os.path.join(root, "masks", "*.npy")))

    train_loader, val_loader = get_single_batch_normal_loader(
        root_images=root_images, root_labels=root_labels, batch_size=8)

    for x, y in train_loader:
        print(f"Image: {x.shape} || Mask: {y.shape}")


def test_single_batch_rotatory():
    root = "../Triple-View-R-Net/data_for_training/MMWHS/"

    root_images = sorted(
        glob(os.path.join(root, "ct_train", "images", "*.nii.gz")))
    root_labels = sorted(
        glob(os.path.join(root, "ct_train", "masks", "*.nii.gz")))

    train_loader, val_loader = get_single_batch_rotatory_loader(
        root_images=root_images, root_labels=root_labels)

    train_transform_trivial = transforms.Compose([
        transforms.TrivialAugmentWide(num_magnitude_bins=5),
    ])

    for x, y in train_loader:

        x = duplicate_open_end(x)
        y = duplicate_open_end(y)

        length = x.shape[-1]

        print(f"Image Volume: {x.shape} || Mask Volume: {y.shape}")

        for i in range(0, length, 7):

            if i + 8 >= length:
                num_slice = length - i
                if num_slice < 3:
                    for i in range(3 - num_slice):
                        x = duplicate_end(x)
                        y = duplicate_end(y)

                    num_slice = 3

            else:
                num_slice = 8

            images, masks = get_slice_from_volumetric_data(
                x, y, i, num_slice)

            print(images.shape, masks.shape)

            break

        print("-" * 30)


def main():
    test_single_batch_rotatory()


if __name__ == "__main__":
    main()
