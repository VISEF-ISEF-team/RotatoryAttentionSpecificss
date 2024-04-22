import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
import nibabel as nib
from torchvision import transforms
from utils import duplicate_end, duplicate_open_end, get_slice_from_volumetric_data


class RotatoryModelDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(self.images_path)

    def __len__(self):
        return self.n_samples

    def normalize_image_intensity_range(self, img):
        HOUNSFIELD_MAX = np.max(img)
        HOUNSFIELD_MIN = np.min(img)
        HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

        img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
        img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX

        return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE

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

        image = self.normalize_image_intensity_range(image)
        self.convert_label_to_class(mask)

        return image, mask


def load_dataset(image_path, mask_path, split=0.2, image_extension="*.nii.gz", mask_extension="*.nii.gz"):
    images = sorted(glob(os.path.join(image_path, image_extension)))
    masks = sorted(glob(os.path.join(mask_path, mask_extension)))

    split_size = int(split * len(images))

    x_train, x_val = train_test_split(
        images, test_size=split_size, random_state=42)

    y_train, y_val = train_test_split(
        masks, test_size=split_size, random_state=42)

    return (x_train, y_train), (x_val, y_val)


def get_loaders(root_images, root_labels, batch_size, num_workers, image_extension, mask_extension):
    (x_train, y_train), (x_val, y_val) = load_dataset(root_images, root_labels,
                                                      image_extension=image_extension, mask_extension=mask_extension)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    train_dataset = RotatoryModelDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = RotatoryModelDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return (train_loader, val_loader)


def main():
    root = "../Triple-View-R-Net/data_for_training/MMWHS/"

    root_images = os.path.join(root, "ct_train", "images")
    root_labels = os.path.join(root, "ct_train", "masks")

    image_extension = "*.nii.gz"
    mask_extension = "*.nii.gz"

    print(root_images)

    train_loader, val_loader = get_loaders(
        root_images, root_labels, batch_size=None, num_workers=6, image_extension=image_extension, mask_extension=mask_extension)

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
                x, y, i, num_slice, train_transform=train_transform_trivial)

            print(images.shape, masks.shape)

        print("-" * 30)


if __name__ == "__main__":
    main()
