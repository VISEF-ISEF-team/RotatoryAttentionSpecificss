import os
from glob import glob
from dataset import get_normal_loaders, get_single_batch_normal_loader, get_rotatory_loaders, get_single_batch_rotatory_loader


def load_MMWHS(batch_size, num_workers, split, rot, test):
    if rot:
        root = "../Triple-View-R-Net/data_for_training/MMWHS/"

        root_images = sorted(
            glob(os.path.join(root, "ct_train", "images", "*.nii.gz")))
        root_labels = sorted(
            glob(os.path.join(root, "ct_train", "masks", "*.nii.gz")))

        if test:
            train_loader, val_loader = get_single_batch_rotatory_loader(
                root_images=root_images, root_labels=root_labels, split=split, num_workers=num_workers)
        else:
            train_loader, val_loader = get_rotatory_loaders(
                root_images=root_images, root_labels=root_labels, split=split, num_workers=num_workers)
    else:
        root = "../Triple-View-R-Net/data_for_training/MMWHS/"

        root_images = sorted(
            glob(os.path.join(root, "images", "*.png")))
        root_labels = sorted(
            glob(os.path.join(root, "masks", "*.npy")))

        if test:
            train_loader, val_loader = get_single_batch_normal_loader(
                root_images=root_images, root_labels=root_labels, batch_size=batch_size, split=split, num_workers=num_workers, num_classes=8)
        else:
            train_loader, val_loader = get_normal_loaders(
                root_images=root_images, root_labels=root_labels, batch_size=batch_size, split=split, num_workers=num_workers, num_classes=8)

    return train_loader, val_loader


def get_dataset(dataset_name, batch_size, num_workers, split, rot, test):
    if dataset_name == "MMWHS":
        train_loader, val_loader = load_MMWHS(
            batch_size, num_workers=num_workers, split=split, rot=rot, test=test)

    return train_loader, val_loader


if __name__ == "__main__":
    batch_size = 20
    num_workers = 8
    split = 0.6
    rot = False

    train_loader, val_loader = get_dataset(
        dataset_name="MMWHS", batch_size=batch_size, num_workers=num_workers, split=split, rot=rot)

    for x, y in train_loader:
        print(f"Image: {x.shape} || Mask: {y.shape}")
