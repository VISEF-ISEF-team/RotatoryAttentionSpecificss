import numpy as np
import nibabel as nib
from glob import glob
from skimage.transform import resize
import cv2
import os


class OneHotEncoder():
    def __init__(self):
        self.unique_values = None
        self.num_classes = None

    def encode(self, label: np.array):
        """
        label: a 2D numpy array that contains raw class values, whether they are class labels or arbitrary values
        """

        output = [np.where(label == x, 1, 0) for x in self.unique_values]
        output = np.stack(output, axis=0)
        return output

    def set_unique_values(self, volumetric_label_path):
        """
        volumetric label containing all the classes that would be present in each slices of the image
        """
        volumetric_array = nib.load(volumetric_label_path).get_fdata()

        self.unique_values = np.unique(volumetric_array)

        self.num_classes = len(self.unique_values)

    def get_num_classes(self):
        return self.num_classes

    def get_unique_values(self):
        return self.unique_values


class VolumetricConverter():
    def __init__(self, root_path):
        self.root_path = root_path

        # get images path
        self.images_path = sorted(
            glob(os.path.join(root_path, "*", "*", "*image*.nii.gz")))

        # get labels path
        self.labels_path = sorted(
            glob(os.path.join(root_path, "*", "*", "*label*.nii.gz")))

        # create directory for storing images and labels
        self.images_folder_path = os.path.join(root_path, "images")
        self.labels_folder_path = os.path.join(root_path, "labels")

        if not os.path.isdir(self.images_folder_path):
            os.mkdir(self.images_folder_path)

        if not os.path.isdir(self.labels_folder_path):
            os.mkdir(self.labels_folder_path)

        assert len(self.images_path) != 0, print(f"No image paths found.")

        assert len(self.labels_path) != 0, print(f"No label paths found.")

        assert len(self.images_path) == len(self.labels_path), print(
            f"Number of images and labels do not match in the dataset.")

        self.n_samples = len(self.images_path)

        # set up encoder for one hot encoding
        self.encoder = OneHotEncoder()
        self.encoder.set_unique_values(self.labels_path[0])

    def normalize_image_intensity_range(self, img):
        HOUNSFIELD_MAX = np.max(img)
        HOUNSFIELD_MIN = np.min(img)
        HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

        img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
        img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX

        return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE

    def save_image(self, img_path: str, nii_index: int, new_size: tuple):
        # load volumetric data
        array = nib.load(img_path).get_fdata()

        # normalize data
        array = self.normalize_image_intensity_range(array)

        # create separate folder for volumetric data
        current_folder_path = os.path.join(
            self.images_folder_path, str(nii_index))

        if not os.path.isdir(current_folder_path):
            os.mkdir(current_folder_path)

        for i in range(array.shape[-1]):
            # get image slice
            img_slice = array[:, :, i]

            # create file name
            img_name = f"heart{nii_index}-slice{str(i).zfill(3)}_axial"

            # create image path
            path = os.path.join(self.images_folder_path,
                                str(nii_index), f"{img_name}.png")

            # resize image
            img = cv2.resize(img_slice, new_size)
            img = np.uint8(img * 255)

            # save image
            res = cv2.imwrite(path, img)

            # guarantee image is saved
            if not res:
                print(f"Error, unable to save image with name: {img_name}")

    def save_label(self, label_path: str, nii_index: int, new_size: tuple):
        # load volumetric label
        output = nib.load(label_path).get_fdata()

        # create separate folder for volumetric data
        current_folder_path = os.path.join(
            self.labels_folder_path, str(nii_index))

        if not os.path.isdir(current_folder_path):
            os.mkdir(current_folder_path)

        for i in range(output.shape[-1]):
            # get label slice
            label = output[:, :, i]

            # create label name
            label_name = f"heartmaskencode{nii_index}-slice{str(i).zfill(3)}_axial"

            # create label path
            path = os.path.join(self.labels_folder_path,
                                str(nii_index), f"{label_name}.npy")

            # use encoder to encode label
            encoded_label = self.encoder.encode(label)

            # resize the encoded label for maximum resize accuracy
            encoded_label_reshape = resize(
                encoded_label, (self.encoder.get_num_classes(), *new_size), preserve_range=True, anti_aliasing=True)

            # argmax to get back class values
            encoded_label_reshape = np.argmax(encoded_label_reshape, axis=0)

            # save label
            np.save(path, encoded_label_reshape)

    def convert_volumetric_to_slice(self, new_size):
        for nii_index in range(self.n_samples):
            image_path = self.images_path[nii_index]
            label_path = self.labels_path[nii_index]

            print(f"Image path: {image_path} || Label path: {label_path}")

            # call saving functions
            self.save_image(image_path, nii_index, new_size=new_size)
            self.save_label(label_path, nii_index, new_size=new_size)

    def get_length(self):
        print(
            f"Image length: {len(self.images_path)} || Labels length: {len(self.labels_path)}")


if __name__ == "__main__":
    root_path = "./data_for_training/MMWHS"

    converter = VolumetricConverter(root_path=root_path)
    encoder = OneHotEncoder()

    converter.convert_volumetric_to_slice((256, 256))
