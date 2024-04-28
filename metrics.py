import numpy as np


def dice_score(y, y_pred):
    intersection = y * y_pred
    smooth = 1e-7
    dice = (2. * intersection + smooth) / (y.sum() + y_pred.sum() + smooth)

    return dice


def multiclass_dice_score(y, y_pred, num_classes, class_weights=None):
    if class_weights != None:
        assert len(class_weights) == num_classes, print(
            f"Class weights has length: {len(class_weights)} not equal to number of classes: {num_classes}")

        total_dice_score = 0.0
        total_weights = sum(class_weights)

        for i in range(num_classes):
            y_bin = np.where(y == i, 1.0, 0.0)
            y_pred_bin = np.where(y_pred == i, 1.0, 0.0)

            dsc = dice_score(y_bin, y_pred_bin) * class_weights[i]

            total_dice_score += dsc

        return total_dice_score / total_weights

    else:
        total_dice_score = 0.0

        for i in range(num_classes):
            y_bin = np.where(y == i, 1.0, 0.0)
            y_pred_bin = np.where(y_pred == i, 1.0, 0.0)

            dsc = dice_score(y_bin, y_pred_bin)

            total_dice_score += dsc

        return total_dice_score / num_classes
