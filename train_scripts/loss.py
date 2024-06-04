import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def torch_dice_score(inputs, targets, smooth=1):
    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth) / \
        (inputs.sum() + targets.sum() + smooth)

    return dice


def soft_dice_score(inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None, weights=None) -> torch.Tensor:
    assert inputs.size() == targets.size(), print(
        f"Input size: {inputs.size()} does not match targets: {targets.size()}")

    if dims is not None:
        intersection = torch.sum(inputs * targets, dim=dims)
        cardinality = torch.sum(inputs + targets, dim=dims)
    else:
        intersection = torch.sum(inputs * targets)
        cardinality = torch.sum(inputs + targets)

    dice_score = (2.0 * intersection + smooth) / \
        (cardinality + smooth).clamp_min(eps)

    if weights != None and dims != None:
        dice_score *= weights

    elif weights != None and dims == None:
        raise RuntimeError(
            "Weights are provided but no dimensoins over classes are provided.")

    return dice_score


def mean_aggregate_score(inputs: torch.Tensor, filter=True, weights=None):
    # if filter:
    #     inputs = inputs[inputs != 0.0]

    if weights != None:
        inputs = inputs.sum() / sum(weights)
    else:
        res = inputs.mean()

    return res


class MulticlassDiceLoss(nn.Module):
    def __init__(self, weight: list = None):
        super().__init__()
        self.weight = weight

        self.score = soft_dice_score
        self.aggregate = mean_aggregate_score

        self.smooth = 0.0
        self.eps = 1e-7

    def forward(self, inputs, targets, multiclass=True, filter=True):
        """Inputs """
        if multiclass:
            dims = (0, 2)
        else:
            dims = None

        # softmax for input
        inputs = inputs.log_softmax(dim=1).exp()

        # flatten input
        inputs = inputs.flatten(start_dim=2)
        targets = targets.flatten(start_dim=2)

        dice_score = self.score(
            inputs, targets, smooth=self.smooth, eps=self.eps, dims=dims, weights=self.weight)

        loss = -torch.log(dice_score.clamp_min(1e-7))
        mask = targets.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        res = self.aggregate(loss, filter=filter, weights=self.weight)

        return res


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
    mask = torch.from_numpy(
        np.load("../Triple-View-R-Net/data_for_training/MMWHS/masks/heartmaskencode0-slice130_axial.npy"))

    y_pred = torch.from_numpy(
        np.load("../Triple-View-R-Net/data_for_training/MMWHS/masks/heartmaskencode0-slice145_axial.npy"))

    print(torch.unique(mask))
    print(torch.unique(y_pred))

    # encode prediction
    y_pred_encode = F.one_hot(y_pred, num_classes=8)
    y_pred_encode = y_pred_encode.permute(2, 0, 1)
    y_pred_encode = torch.unsqueeze(y_pred_encode, dim=0)
    print(f"Y pred encode: {y_pred_encode.shape}")

    # encode mask
    mask_encode = F.one_hot(mask, num_classes=8)
    mask_encode = mask_encode.permute(2, 0, 1)
    mask_encode = torch.unsqueeze(mask_encode, dim=0)
    print(f"Mask encode: {mask_encode.shape}")

    # y_pred_encode = y_pred_encode.flatten(start_dim=2)
    # mask_encode = mask_encode.flatten(start_dim=2)
    # print(f"Y pred encode flatten: {y_pred_encode.shape}")
    # print(f"Mask encode flatten: {mask_encode.shape}")

    # dims = (0, 2)
    # intersection = torch.sum(y_pred_encode * mask_encode, dim=dims)
    # cardinality = torch.sum(y_pred_encode + mask_encode, dim=dims)

    # dice_score = (2.0 * intersection + 0) / \
    #     (cardinality + 0).clamp_min(1e-7)

    # print(dice_score)

    # loss = -torch.log(dice_score.clamp_min(1e-7))

    # print(loss)

    # mask = mask_encode.sum(dims) > 0
    # loss *= mask.to(loss.dtype)

    # print(loss)

    # loss = loss[loss != 0.0]
    # res = loss.mean()
    # print(res)

    # dice_score_filtered = dice_score[dice_score != 0.0]

    # print(dice_score_filtered)

    # dice_score_mean = torch.mean(dice_score_filtered)
    # print(dice_score_mean)

    loss_fn = MulticlassDiceLoss()
    loss = loss_fn(y_pred_encode.float(), mask_encode.float())

    print(f"Multiclass: {loss.item()}")
