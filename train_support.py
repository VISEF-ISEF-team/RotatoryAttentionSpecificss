import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from skimage.transform import resize
import torch.nn.functional as F
from loss import CustomDiceLoss
from networks.original_unet_attention import Attention_Unet
from networks.rotatory_attention_unet import Rotatory_Attention_Unet
from networks.rotatory_attention_unet_v2 import Rotatory_Attention_Unet_v2
from networks.rotatory_attention_unet_v3 import Rotatory_Attention_Unet_v3
from networks.unet import Unet


def get_optimziers(optimizer_name, parameters, learning_rate):
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params=parameters, lr=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params=parameters, lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(params=parameters, lr=learning_rate)

    return optimizer


def get_loss_fn(loss_fn_name):
    if loss_fn_name == "dice":
        loss = CustomDiceLoss()
    else:
        loss = CustomDiceLoss()
    return loss


def get_models(model_name, num_classes=8, image_size=128):
    # "unet", "rotatory_unet_attention", "rotatory_unet_attention_v3", "vit", "unetmer"
    if model_name == "unet_attention":
        model = Attention_Unet(num_classes=num_classes)
    elif model_name == "unet":
        model = Unet(outc=num_classes)
    elif model_name == "rotatory_unet_attention":
        model = Rotatory_Attention_Unet(image_size=image_size)
    elif model_name == "rotatory_unet_attention_v2":
        model = Rotatory_Attention_Unet_v2(image_size=image_size)
    elif model_name == "rotatory_unet_attention_v3":
        model = Rotatory_Attention_Unet_v3(image_size=image_size)

    return model


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
