import time
import torch
from tqdm import tqdm
import torch.nn as nn
from monai.losses import DiceLoss, TverskyLoss
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score
import pandas as pd
from train_scripts.metrics import multiclass_dice_score
import numpy as np


def normal_train(model, loader, optimizer, loss_fn, num_classes, scaler, device=torch.device("cuda")):
    model.train()
    pbar = tqdm(loader)
    total_steps = len(loader)
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_dice_coef = 0.0
    epoch_jaccard = 0.0
    epoch_recall = 0.0
    epoch_f1 = 0.0

    for step, (x, y) in enumerate(pbar):
        pbar.set_description(
            f"Train step: {step} / {total_steps}")
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # forward:
        y_pred = model(x)

        y = nn.functional.one_hot(y.long(), num_classes=num_classes)
        y = torch.squeeze(y, dim=1)
        y = y.permute(0, 3, 1, 2)

        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        """Take argmax for accuracy calculation"""
        y_pred = y_pred.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        """Update batch metrics"""
        batch_dice_coef = multiclass_dice_score(
            y=y, y_pred=y_pred)

        # take argmax
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)

        batch_accuracy = accuracy_score(
            y.flatten(), y_pred.flatten())

        batch_jaccard = jaccard_score(
            y.flatten(), y_pred.flatten(), average="micro")

        batch_recall = recall_score(
            y.flatten(), y_pred.flatten(), average="micro")

        batch_f1 = f1_score(y.flatten(),
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

    epoch_loss = epoch_loss / total_steps
    epoch_dice_coef = epoch_dice_coef / total_steps
    epoch_accuracy = epoch_accuracy / total_steps
    epoch_jaccard = epoch_jaccard / total_steps
    epoch_recall = epoch_recall / total_steps
    epoch_f1 = epoch_f1 / total_steps

    return epoch_loss, epoch_dice_coef, epoch_accuracy, epoch_jaccard, epoch_recall, epoch_f1


def normal_evaluate(model, loader, loss_fn, num_classes, scaler, device=torch.device("cuda")):
    model.eval()
    pbar = tqdm(loader)
    total_steps = len(loader)
    epoch_loss = 0.0
    epoch_f1 = 0.0
    epoch_accuracy = 0.0
    epoch_recall = 0.0
    epoch_jaccard = 0.0
    epoch_dice_coef = 0.0

    with torch.no_grad():
        for step, (x, y) in enumerate(pbar):
            pbar.set_description(f"Validation step: {step} / {total_steps}")
            x = x.to(device)
            y = y.to(device)

            """Forward pass"""
            y_pred = model(x)

            y = torch.squeeze(y, dim=1)
            y = nn.functional.one_hot(y.long(), num_classes=num_classes)
            y = y.permute(0, 3, 1, 2)

            """Calculate loss"""
            loss = loss_fn(y_pred, y)

            """Convert to numpy for metrics calculation"""
            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            """Batch metrics calculation"""
            batch_dice_coef = multiclass_dice_score(
                y=y, y_pred=y_pred)

            # take argmax
            y_pred = np.argmax(y_pred, axis=1)
            y = np.argmax(y, axis=1)

            batch_loss = loss.item()
            batch_f1 = f1_score(y.flatten(),
                                y_pred.flatten(), average="micro")
            batch_accuracy = accuracy_score(
                y.flatten(), y_pred.flatten())
            batch_recall = recall_score(
                y.flatten(), y_pred.flatten(), average="micro")
            batch_jaccard = jaccard_score(
                y.flatten(), y_pred.flatten(), average="micro")

            """Epoch metrics calculation"""
            epoch_loss += batch_loss
            epoch_f1 += batch_f1
            epoch_accuracy += batch_accuracy
            epoch_recall += batch_recall
            epoch_jaccard += batch_jaccard
            epoch_dice_coef += batch_dice_coef

            pbar.set_postfix(
                {"loss": batch_loss, "dice_coef": batch_dice_coef, "accuracy": batch_accuracy, "iou": batch_jaccard})

    epoch_loss = epoch_loss / total_steps
    epoch_f1 = epoch_f1 / total_steps
    epoch_accuracy = epoch_accuracy / total_steps
    epoch_recall = epoch_recall / total_steps
    epoch_jaccard = epoch_jaccard / total_steps
    epoch_dice_coef = epoch_dice_coef / total_steps

    return epoch_loss, epoch_f1, epoch_accuracy, epoch_recall, epoch_jaccard, epoch_dice_coef
