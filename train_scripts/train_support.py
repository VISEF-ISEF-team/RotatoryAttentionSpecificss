import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from skimage.transform import resize
import torch.nn.functional as F
from train_scripts.loss import CustomDiceLoss, MulticlassDiceLoss


def get_optimziers(optimizer_name, parameters, learning_rate):
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params=parameters, lr=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params=parameters, lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params=parameters, lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(params=parameters, lr=learning_rate)

    return optimizer


def get_loss_fn(loss_fn_name):
    if loss_fn_name == "dice":
        loss = CustomDiceLoss()
    elif loss_fn_name == "multi_dice":
        weights = None
        loss = MulticlassDiceLoss(weight=weights)

    return loss


def normalize_image_intensity_range(img):
    HOUNSFIELD_MAX = np.max(img)
    HOUNSFIELD_MIN = np.min(img)
    HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX

    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE


def get_slice_from_volumetric_data(image_volume, mask_volume, start_idx, num_slice=12, train_transform=None, val_transform=None):

    end_idx = start_idx + num_slice

    images = torch.empty(num_slice, 1, 256, 256)
    masks = torch.empty(num_slice, 1, 256, 256)

    for i in range(start_idx, end_idx, 1):
        image = image_volume[:, :, i].numpy()
        image = cv2.resize(image, (256, 256))
        image = normalize_image_intensity_range(image).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)

        if train_transform != None:
            image = train_transform(image)

        if val_transform != None:
            image = val_transform(image)

        images[i - start_idx, :, :, :] = image

        mask = mask_volume[:, :, i].long()
        mask = F.one_hot(mask, num_classes=8)
        mask = mask.numpy()
        mask = resize(mask, (256, 256, 8),
                      preserve_range=True, anti_aliasing=True)
        mask = torch.from_numpy(mask)
        mask = torch.argmax(mask, dim=-1)
        mask = torch.unsqueeze(mask, dim=0)

        masks[i - start_idx, :, :, :] = mask

    return images, masks


def duplicate_open_end(x):
    first_slice = x[:, :, 0].unsqueeze(2)
    last_slice = x[:, :, -1].unsqueeze(2)
    x = torch.cat((first_slice, x, last_slice), dim=2)

    return x


def duplicate_end(x):
    last_slice = x[:, :, -1].unsqueeze(2)
    x = torch.cat((x, last_slice), dim=2)

    return x


def get_new_batch_size(length, batch_size):
    optimal_batch_size = 0

    for i in range(batch_size, 3 - 1, -1):
        if length % i >= 3:
            optimal_batch_size = max(optimal_batch_size, i)

    return optimal_batch_size
