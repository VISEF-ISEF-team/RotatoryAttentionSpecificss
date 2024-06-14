import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
from glob import glob

if __name__ == "__main__":
    path = sorted(glob(os.path.join("data_for_training",
                  "ImageTBAD", "imageTBAD", "*label.nii.gz")))

    for label in path:
        a = nib.load(label).get_fdata()

        print(
            f"Unique values: {np.unique(a)} || Num classes: {len(np.unique(a))}")
