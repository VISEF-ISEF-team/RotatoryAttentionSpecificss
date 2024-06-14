import os
from glob import glob
from data_loaders import get_normal_loaders, get_single_batch_normal_loader, get_rotatory_loaders, get_single_batch_rotatory_loader


def get_dataset_loaders(dataset_name, batch_size, num_workers, split, rot, test, num_classes):
    if rot:
        root = f"./data_for_training/{dataset_name}/"

        root_images = sorted(
            glob(os.path.join(root, "images", "*")))
        root_labels = sorted(
            glob(os.path.join(root, "labels", "*")))

        if test:
            train_loader, val_loader = get_single_batch_rotatory_loader(
                root_images=root_images, root_labels=root_labels, batch_size=batch_size, num_classes=num_classes, split=split, num_workers=num_workers)
        else:
            train_loader, val_loader = get_rotatory_loaders(
                root_images=root_images, root_labels=root_labels, batch_size=batch_size, num_classes=num_classes, split=split, num_workers=num_workers)
    else:
        root = f"./data_for_training/{dataset_name}/"

        root_images = sorted(
            glob(os.path.join(root, "images", "*",  "*.png")))
        root_labels = sorted(
            glob(os.path.join(root, "labels", "*", "*.npy")))

        if test:
            train_loader, val_loader = get_single_batch_normal_loader(
                root_images=root_images, root_labels=root_labels, batch_size=batch_size, num_classes=num_classes, split=split, num_workers=num_workers)
        else:
            train_loader, val_loader = get_normal_loaders(
                root_images=root_images, root_labels=root_labels, batch_size=batch_size, num_classes=num_classes, split=split, num_workers=num_workers)

    return train_loader, val_loader


def get_dataset(dataset_name, batch_size, num_workers, split, rot, test=False):
    if dataset_name == "MMWHS":
        num_classes = 8
        color_channel = 1
        train_loader, val_loader = get_dataset_loaders(
            dataset_name="MMWHS", batch_size=batch_size, num_workers=num_workers, split=split, rot=rot, test=test, num_classes=num_classes)

    elif dataset_name == "Synapse":
        num_classes = 14
        color_channel = 1

        train_loader, val_loader = get_dataset_loaders(
            dataset_name="Synapse", batch_size=batch_size, num_workers=num_workers, split=split, rot=rot, test=test, num_classes=num_classes)

    elif dataset_name == "ImageTBAD":
        num_classes = 4
        color_channel = 1

        train_loader, val_loader = get_dataset_loaders(
            dataset_name="ImageTBAD", batch_size=batch_size, num_workers=num_workers, split=split, rot=rot, test=test, num_classes=num_classes)

    return color_channel, num_classes, train_loader, val_loader


if __name__ == "__main__":
    batch_size = 20
    num_workers = 8
    split = 0.2
    rot = True

    color_channel, num_classes, train_loader, val_loader = get_dataset(
        dataset_name="ImageTBAD", batch_size=batch_size, num_workers=num_workers, split=split, rot=rot, test=False)

    counter = 0

    for x, y in train_loader:
        batch_size = x.shape[0]

        if batch_size < 3:
            print(f"Error, batch size < 3")
            break

        print(
            f"Batch size: {batch_size} || Image: {x.shape} || Mask: {y.shape}")
        counter += 1

    print(f"Counter: {counter}")
