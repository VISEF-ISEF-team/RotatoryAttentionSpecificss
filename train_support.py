import torch
import torch.nn as nn
import os
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
        optimzier = torch.optim.AdamW(params=parameters, lr=learning_rate)
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


def get_dataset(dataset_name, rot):
    if dataset_name == "MMWHS":
        if rot:
            root = "../Triple-View-R-Net/data_for_training/MMWHS/"

            root_images = os.path.join(root, "ct_train", "images")
            root_masks = os.path.join(root, "ct_train", "masks")

            image_extension = "*.nii.gz"
            mask_extension = "*.nii.gz"

        else:
            root_images = "../Triple-View R-Net/data_for_training/MMWHS/images/"
            root_masks = "../Triple-View R-Net/data_for_training/MMWHS/masks/"
            image_extension = "*.png"
            mask_extension = "*.npy"

    return root_images, root_masks, image_extension, mask_extension
