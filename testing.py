import time
from glob import glob
import os
import numpy as np


if __name__ == "__main__":
    a = np.load(
        "./data_for_training/MMWHS/labels/0/heartmaskencode0-slice090_axial.npy")

    print(np.unique(a), np.min(a), np.max(a))
