import datetime
import csv
import torch
import cv2
import numpy as np
from skimage.transform import resize
import os
import torch.nn.functional as F
import random


def set_seeds():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # Set the seed for PyTorch's random number generators
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seconds_to_hms(seconds):
    time_obj = datetime.timedelta(seconds=seconds)
    return str(time_obj)


def write_csv(path, data, first=False):
    if first:
        with open(path, mode='w', newline='') as file:
            iteration = csv.writer(file)
            iteration.writerow(data)
        file.close()

    else:
        with open(path, mode='a', newline='') as file:
            iteration = csv.writer(file)
            iteration.writerow(data)
        file.close()


def get_slice_from_volumetric_data(image_volume, mask_volume, start_idx, num_slice=12, train_transform=None, val_transform=None):

    end_idx = start_idx + num_slice

    images = torch.empty(num_slice, 1, 256, 256)
    masks = torch.empty(num_slice, 1, 256, 256)

    for i in range(start_idx, end_idx, 1):
        image = image_volume[:, :, i].numpy()
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.uint8)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)

        if train_transform != None:
            image = train_transform(image)

        elif val_transform != None:
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


def check_directory_exists(directory):
    if os.path.isdir(directory):
        return
    else:
        os.mkdir(directory)


def write_hyperparameters(directory_path, data: dict):
    with open(os.path.join(directory_path, "parameters_log.txt")) as f:
        for key, value in data.items():
            f.write(f"{key}: {data}\n")
