import numpy as np
import torch


def dice_score(y, y_pred):
    intersection = np.sum(y * y_pred)
    smooth = 1e-7
    dice = (2. * intersection + smooth) / (y.sum() + y_pred.sum() + smooth)

    return dice


def multiclass_dice_score(y: np.ndarray, y_pred: np.ndarray, smooth: float = 0.0, eps: float = 1e-7, dims=(0, 2)) -> torch.Tensor:
    """
    Args: 
    y: one hot encoded numpy array as inputs for calculating dice score 
    y_pred: one hot encoded numpy array as mask 
    """

    assert y.size == y_pred.size, print(
        f"Input size: {y.size} does not match targets: {y_pred.size}")

    # flatten
    y = y.reshape(*y.shape[:-2], -1)
    y_pred = y_pred.reshape(*y_pred.shape[:-2], -1)

    intersection = np.sum(y * y_pred, axis=dims)
    cardinality = np.sum(y + y_pred, axis=dims)

    dice_score = (2.0 * intersection + smooth) / \
        (cardinality + smooth).clip(min=eps)

    mask = y_pred.sum(dims) > 0
    dice_score *= mask

    # dice_score_filtered = dice_score[dice_score != 0.0]
    # dice_score_filtered = np.mean(dice_score_filtered)

    dice_score = np.mean(dice_score)

    return dice_score
