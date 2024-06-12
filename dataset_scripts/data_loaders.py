import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import RotatoryModelDataset, NormalModelDataset
from batch_sampler import RotatoryBatchSampler


def load_dataset(images, masks, split=0.2):
    split_size = int(split * len(images))

    x_train, x_val = train_test_split(
        images, test_size=split_size, random_state=42)

    y_train, y_val = train_test_split(
        masks, test_size=split_size, random_state=42)

    return (x_train, y_train), (x_val, y_val)


def get_rotatory_loaders(root_images, root_labels, batch_size, num_classes, split=0.2, num_workers=8):

    # split load
    (x_train, y_train), (x_val, y_val) = load_dataset(
        root_images, root_labels, split=split)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    # load datasets
    train_dataset = RotatoryModelDataset(
        images_folder=x_train, labels_folder=y_train, num_classes=num_classes)

    val_dataset = RotatoryModelDataset(
        images_folder=x_val, labels_folder=y_val, num_classes=num_classes)

    # load samplers
    train_batch_sampler = RotatoryBatchSampler(
        samples=train_dataset.samples, batch_size=batch_size, drop_last=False)

    val_batch_sampler = RotatoryBatchSampler(
        samples=val_dataset.samples, batch_size=batch_size, drop_last=False)

    # load loaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers)

    val_loader = DataLoader(
        dataset=val_dataset, batch_sampler=val_batch_sampler, num_workers=num_workers)

    return (train_loader, val_loader)


def get_normal_loaders(root_images, root_labels, batch_size, num_classes, split=0.2, num_workers=8):

    (x_train, y_train), (x_val, y_val) = load_dataset(
        root_images, root_labels, split=split)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    train_dataset = NormalModelDataset(
        x_train, y_train, num_classes=num_classes)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = NormalModelDataset(x_val, y_val, num_classes=num_classes)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return (train_loader, val_loader)


def get_single_batch_normal_loader(root_images, root_labels, batch_size, num_classes, split=0.2, num_workers=6):
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


def get_single_batch_rotatory_loader(root_images, root_labels, batch_size, num_classes, split=0.2, num_workers=6):
    (x_train, y_train), (x_val, y_val) = load_dataset(
        root_images, root_labels, split=split)

    random_train_index = np.random.randint(low=0, high=len(x_train))
    random_val_index = np.random.randint(low=0, high=len(x_val))

    x_train = [x_train[random_train_index]]
    y_train = [y_train[random_train_index]]

    x_val = [x_val[random_val_index]]
    y_val = [y_val[random_val_index]]

    train_dataset = RotatoryModelDataset(
        images_folder=x_train, labels_folder=y_train, num_classes=num_classes)

    val_dataset = RotatoryModelDataset(
        images_folder=x_val, labels_folder=y_val, num_classes=num_classes)

    # load samplers
    train_batch_sampler = RotatoryBatchSampler(
        samples=train_dataset.samples, batch_size=batch_size, drop_last=False)

    val_batch_sampler = RotatoryBatchSampler(
        samples=val_dataset.samples, batch_size=batch_size, drop_last=False)

    # load loaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers)

    val_loader = DataLoader(
        dataset=val_dataset, batch_sampler=val_batch_sampler, num_workers=num_workers)

    return (train_loader, val_loader)
