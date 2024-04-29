import torch
import torch.nn as nn
import numpy as np


def torch_dice_score(inputs, targets, smooth=1):
    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth) / \
        (inputs.sum() + targets.sum() + smooth)

    return dice


class MulticlassDiceLoss(nn.Module):
    def __init__(self, weight: list = None):
        super().__init__()
        self.weight = weight
        self.score = torch_dice_score

    def forward(self, inputs, targets, smooth=1e-7):
        num_classes = inputs.shape[1]

        inputs_maxed = inputs.argmax(dim=1)
        targets_maxed = targets.argmax(dim=1)

        if self.weight != None:
            assert num_classes == len(self.weight), print(
                f"Weight has length: {len(self.weight)} not equal to num classes: {num_classes}")

            total_weights = sum(self.weight)
            total_dice_score = 0.0

            for i in range(num_classes):
                y_bin = torch.where(inputs_maxed == i, 1.0,
                                    0.0).requires_grad_()
                y_pred_bin = torch.where(
                    targets_maxed == i, 1.0, 0.0).requires_grad_()

                dsc = self.score(y_pred_bin, y_bin) * self.weight[i]

                total_dice_score += dsc

            return 1.0 - (total_dice_score / total_weights)

        else:
            total_dice_score = 0.0

            for i in range(num_classes):
                y_bin = torch.where(inputs_maxed == i, 1.0,
                                    0.0).requires_grad_()
                y_pred_bin = torch.where(
                    targets_maxed == i, 1.0, 0.0).requires_grad_()

                dsc = torch_dice_score(y_pred_bin, y_bin)

                total_dice_score += dsc

            return 1.0 - (total_dice_score / num_classes)


class CustomDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomDiceLoss, self).__init__()
        self.score = torch_dice_score

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.softmax(inputs, dim=1)

        # flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        dsc = self.score(inputs, targets)

        return 1.0 - dsc


if __name__ == "__main__":
    loss = MulticlassDiceLoss()
    y = torch.randn(1, 12, 256, 256)
    y_pred = torch.randn(1, 12, 256, 256)

    loss_item = loss(y_pred, y)

    print(loss_item)
