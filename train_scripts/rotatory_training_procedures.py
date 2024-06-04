import time
import torch
from tqdm import tqdm
import torch.nn as nn
from monai.losses import DiceLoss, TverskyLoss
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score
import pandas as pd
from train_support import duplicate_open_end, get_slice_from_volumetric_data, duplicate_end, get_new_batch_size
from train_scripts.metrics import multiclass_dice_score
import numpy as np


def rotatory_train(model, loader, optimizer, loss_fn, num_classes, test, scaler, batch_size, device=torch.device("cuda")):
    model.train()
    pbar = tqdm(loader)
    steps = len(loader)
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_dice_coef = 0.0
    epoch_jaccard = 0.0
    epoch_recall = 0.0
    epoch_f1 = 0.0
    iter_counter = 0

    for step, (x, y) in enumerate(pbar):
        # x = duplicate_open_end(x)
        # y = duplicate_open_end(y)

        length = x.shape[-1]

        new_batch_size = get_new_batch_size(length, batch_size)

        # create a custom dataset right here with the batch size for better performance ?

        for i in range(0, length, new_batch_size):
            pbar.set_description(
                f"Iter: {iter_counter} - Step: {step} / {steps}")
            iter_counter += 1

            # ensure balance slice count
            if i + batch_size >= length:
                num_slice = length - i

                if num_slice < 3:
                    for _ in range(3 - num_slice):
                        x = duplicate_end(x)
                        y = duplicate_end(y)

                    num_slice = 3

            else:
                num_slice = batch_size

            x_, y_ = get_slice_from_volumetric_data(
                x, y, i, num_slice, train_transform=None)
            x_ = x_.to(device)
            y_ = y_.to(device)

            optimizer.zero_grad()

            # forward:
            y_pred = model(x_)

            y_ = nn.functional.one_hot(y_.long(), num_classes=num_classes)
            y_ = torch.squeeze(y_, dim=1)
            y_ = y_.permute(0, 3, 1, 2)

            loss = loss_fn(y_pred, y_)

            loss.backward()
            optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            """Detach and convert to numpy array"""
            y_pred = y_pred.detach().cpu().numpy()
            y_ = y_.detach().cpu().numpy()

            """Update batch metrics"""
            batch_dice_coef = multiclass_dice_score(
                y=y_, y_pred=y_pred)

            # take argmax
            y_pred = np.argmax(y_pred, axis=1)
            y_ = np.argmax(y_, axis=1)

            batch_accuracy = accuracy_score(
                y_.flatten(), y_pred.flatten())

            batch_jaccard = jaccard_score(
                y_.flatten(), y_pred.flatten(), average="micro")

            batch_recall = recall_score(
                y_.flatten(), y_pred.flatten(), average="micro")

            batch_f1 = f1_score(y_.flatten(),
                                y_pred.flatten(), average="micro")

            batch_loss = loss.item()

            """Update epoch metrics"""
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            epoch_jaccard += batch_jaccard
            epoch_recall += batch_recall
            epoch_f1 += batch_f1
            epoch_dice_coef += batch_dice_coef

            """Set loop postfix"""
            pbar.set_postfix(
                {"loss": batch_loss, "dice_coef": batch_dice_coef, "accuracy": batch_accuracy, "iou": batch_jaccard})

            if test:
                break

    epoch_loss = epoch_loss / iter_counter
    epoch_dice_coef = epoch_dice_coef / iter_counter
    epoch_accuracy = epoch_accuracy / iter_counter
    epoch_jaccard = epoch_jaccard / iter_counter
    epoch_recall = epoch_recall / iter_counter
    epoch_f1 = epoch_f1 / iter_counter

    return epoch_loss, epoch_dice_coef, epoch_accuracy, epoch_jaccard, epoch_recall, epoch_f1


def rotatory_evaluate(model, loader, loss_fn, batch_size, test, num_classes, device=torch.device("cuda")):

    epoch_loss = 0.0
    epoch_f1 = 0.0
    epoch_accuracy = 0.0
    epoch_recall = 0.0
    epoch_jaccard = 0.0
    epoch_dice_coef = 0.0
    iter_counter = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            # x = duplicate_open_end(x)
            # y = duplicate_open_end(y)

            length = x.shape[-1]

            for i in range(0, length, batch_size):
                iter_counter += 1

                # ensure balance slice count
                if i + batch_size >= length:
                    num_slice = length - i

                    if num_slice < 3:
                        for _ in range(3 - num_slice):
                            x = duplicate_end(x)
                            y = duplicate_end(y)

                        num_slice = 3
                else:
                    num_slice = batch_size

                x_, y_ = get_slice_from_volumetric_data(x, y, i, num_slice)

                x_ = x_.to(device)
                y_ = y_.to(device)

                # pass input through model
                y_pred = model(x_)

                y_ = torch.squeeze(y_, dim=1)
                y_ = nn.functional.one_hot(y_.long(), num_classes=num_classes)
                y_ = y_.permute(0, 3, 1, 2)

                # calculate loss
                loss = loss_fn(y_pred, y_)
                epoch_loss += loss.item()

                # argmax & convert to numpy to calculate metrics
                y_pred = y_pred.detach().cpu().numpy()
                y_ = y_.detach().cpu().numpy()

                # batch metrics calculation
                epoch_dice_coef += multiclass_dice_score(
                    y=y_, y_pred=y_pred)

                # argmax
                y_pred = np.argmax(y_pred, axis=1)
                y_ = np.argmax(y_, axis=1)

                epoch_f1 += f1_score(y_.flatten(),
                                     y_pred.flatten(), average="micro")
                epoch_accuracy += accuracy_score(
                    y_.flatten(), y_pred.flatten())
                epoch_recall += recall_score(
                    y_.flatten(), y_pred.flatten(), average="micro")
                epoch_jaccard += jaccard_score(
                    y_.flatten(), y_pred.flatten(), average="micro")

                if test:
                    break

        epoch_loss = epoch_loss/iter_counter
        epoch_f1 = epoch_f1 / iter_counter
        epoch_accuracy = epoch_accuracy / iter_counter
        epoch_recall = epoch_recall / iter_counter
        epoch_jaccard = epoch_jaccard / iter_counter
        epoch_dice_coef = epoch_dice_coef / iter_counter

    return epoch_loss, epoch_f1, epoch_accuracy, epoch_recall, epoch_jaccard, epoch_dice_coef
