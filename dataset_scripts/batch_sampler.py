import torch
import os
from glob import glob
import cv2
import numpy as np
import math
from torch.utils.data import BatchSampler
from dataset_support import get_new_batch_size


class RotatoryBatchSampler(BatchSampler):
    def __init__(self, samples, batch_size, drop_last):
        # data source is
        self.samples = samples
        self.batch_size = batch_size
        self.drop_last = drop_last

        # data container
        self.data_source = {}

        for sample in self.samples:
            order = sample["order"]
            index = sample["index"]

            if order in self.data_source:
                self.data_source[order].append(index)
            else:
                self.data_source[order] = [index]

        self.total = 0

        for order, indexes in self.data_source.items():
            self.total += math.ceil(len(indexes) / self.batch_size)

    def __len__(self):
        return self.total

    def __iter__(self):
        # can also shuffle the list order
        batch = []

        for order, indexes in self.data_source.items():

            # order: order of 3D volumetric data
            # indexes: list of index belonging to that order

            count = len(indexes)

            # calculate new batch size for rotatory guarantee
            optimal_batch_size = get_new_batch_size(
                length=count, batch_size=self.batch_size)

            # yield function for creating batches
            for i, idx in enumerate(indexes):
                batch.append(idx)

                if i == count - 1 or len(batch) == optimal_batch_size:
                    yield batch
                    batch = []
