import datetime
import csv
import torch
import cv2
import numpy as np
from skimage.transform import resize
import os
import torch.nn.functional as F
import random
import json
from glob import glob


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


def check_directory_exists(directory):
    if os.path.isdir(directory):
        return False
    else:
        os.mkdir(directory)
        return True


def write_hyperparameters(directory_path, data: dict):
    fullpath = os.path.join(directory_path, "hyperparameters_log.json")
    json_object = json.dumps(data, indent=4)

    with open(fullpath, "w") as outfile:
        outfile.write(json_object)


def create_dir_with_id(base_name: str):
    """Base folder name for model training"""
    num_id = len(glob(os.path.join(base_name, "*")))

    while not check_directory_exists(f"{base_name}_{num_id}"):
        num_id += 1

    return num_id


def rename_str_dir(base_name):
    dir_list = glob(os.path.join(base_name, "*"))

    for i, dir in enumerate(dir_list):
        # name: model_name_asd;fla;lsdkf_id
        new_name = dir[:-2] + f"_{str(i)}"

        os.rename(dir, new_name)


def get_device():
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    return device
