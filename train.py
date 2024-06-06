import time
import os
import torch
from utils import seconds_to_hms, write_csv, write_hyperparameters, set_seeds, create_dir_with_id, get_device, rename_str_dir, check_directory_exists
from train_scripts.train_support import get_loss_fn, get_optimziers
from train_scripts.normal_training_procedures import normal_evaluate, normal_train
from train_scripts.rotatory_training_procedures import rotatory_evaluate, rotatory_train
from dataset_scripts.dataset_main_loaders import get_dataset
from model_scripts.model_support import get_models


def total_train_procedure(model_name, dataset_name, optimizer_name, loss_fn_name, num_epochs, batch_size, num_workers=6, image_size=256, rotatory=True, test=False, mixed_precision=False, load_model=False, starting_epoch=0, starting_lr=1e-3):

    set_seeds()

    """Define hyper parameters"""
    lr = starting_lr
    device = get_device()
    # device = torch.device("cpu")

    """Define paths using templates"""
    # get base folder name without id
    base_name = f"./storage/{model_name}_optim-{optimizer_name}_image-{image_size}_lr-{starting_lr}_mixed-{mixed_precision}"

    # get id and define path
    num_id = create_dir_with_id(base_name=base_name)
    directory_path = f"{base_name}_{num_id}"

    # check if num_id is valid & create directory
    check_directory_exists(directory_path)

    # define path for checkpoint files
    train_metrics_path = os.path.join(directory_path, "train_metrics.csv")
    test_metrics_path = os.path.join(directory_path, "test_metrics.csv")
    checkpoint_path = os.path.join(directory_path, "checkpoint.pth.tar")

    """Write params to file"""
    write_hyperparameters(directory_path, data={
                          "Model name": model_name, "Id": num_id, "Train metrics path": train_metrics_path, "Test metrics path": test_metrics_path, "Checkpoint path": checkpoint_path, "Optimizer": optimizer_name, "Epochs": num_epochs, "Starting lr": lr, "Workers": num_workers, "Batch size": batch_size})

    """Initial write to csv to set rows"""
    if not load_model or starting_epoch == 0:
        write_csv(train_metrics_path, ["Epoch", "LR", "Loss", "Dice",
                                       "Accuracy", "Jaccard", "Recall", "F1", "Hausdorff"], first=True)

        write_csv(test_metrics_path, ["Epoch", "LR", "Loss", "Dice",
                                      "Accuracy", "Jaccard", "Recall", "F1", "Hausdorff"], first=True)

    """Get loaderes and dataset information"""
    split = 0.2
    num_classes, train_loader, val_loader = get_dataset(
        dataset_name=dataset_name, batch_size=batch_size, num_workers=num_workers, split=split, rot=rotatory, test=test)

    """Initialize model and more"""
    if load_model:
        checkpoint = torch.load(checkpoint_path)
        model = get_models(model_name=model_name,
                           num_classes=num_classes, image_size=image_size)
        model.load_state_dict(checkpoint)
        model.to(device)
    else:
        model = get_models(model_name=model_name,
                           num_classes=num_classes, image_size=image_size)
        model.to(device)

    """Define optimizer and scheduler"""
    optimizer = get_optimziers(
        optimizer_name=optimizer_name, parameters=model.parameters(), learning_rate=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=3)

    """Define loss function"""
    loss_fn = get_loss_fn(loss_fn_name=loss_fn_name)

    """Define scalar in case of mixed precision"""
    scaler = torch.cuda.amp.GradScaler()

    """Test model output"""
    r = model(torch.rand(6, 1, image_size, image_size).to(device))
    print(f"Testing model output: {r.shape}")

    """ Training the model """
    best_valid_loss = float("inf")

    ############################################################################################################

    for epoch in range(starting_epoch, num_epochs, 1):
        start_time = time.time()

        """Main function call"""

        if rotatory:
            # train function
            train_loss, train_dice_coef, train_accuracy, train_jaccard, train_recall, train_f1, train_hausdorff = rotatory_train(
                model, train_loader, optimizer, loss_fn, test, scaler, device)

            # validate function
            valid_loss, valid_f1, valid_accuracy, valid_recall, valid_jaccard, valid_hausdorff = rotatory_evaluate(
                model, val_loader, loss_fn, test, device)
        else:
            # train function
            train_loss, train_dice_coef, train_accuracy, train_jaccard, train_recall, train_f1, train_hausdorff = normal_train(
                model, train_loader, optimizer, loss_fn, scaler, device)

            # validate function
            valid_loss, valid_f1, valid_accuracy, valid_recall, valid_jaccard, valid_dice_coef, valid_hausdorff = normal_evaluate(
                model, val_loader, loss_fn, scaler, device)

        """WRite to CSV"""
        # write to train
        write_csv(train_metrics_path, [epoch, lr, train_loss, train_dice_coef, train_accuracy,
                                       train_jaccard, train_recall, train_f1, train_hausdorff])

        # write to test
        write_csv(test_metrics_path, [epoch, lr, valid_loss, valid_dice_coef, valid_accuracy,
                                      valid_jaccard, valid_recall, valid_f1, valid_hausdorff])

        """Check loss and update learning rate"""
        scheduler.step(round(valid_loss, 4))

        """Saving and checking loss of model"""
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint at: {checkpoint_path}"
            print(data_str)
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        """Calculate total time"""
        end_time = time.time()
        total_seconds = end_time - start_time
        formatted_time = seconds_to_hms(total_seconds)

        """Format string for printing"""
        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {formatted_time}\n'
        data_str += f'\t LR: {lr} change to {scheduler.get_last_lr()}\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        data_str += f'\t Val. F1: {valid_f1:.3f}\n'
        data_str += f'\t Val. Accuracy: {valid_accuracy:.3f}\n'
        data_str += f'\t Val. Recall: {valid_recall:.3f}\n'
        data_str += f'\t Val. Jaccard: {valid_jaccard:.3f}\n'
        data_str += f'\t Val. Dice Coef: {valid_dice_coef:.3f}\n'
        data_str += f'\t Val. Hausdorff: {valid_dice_coef:.3f}\n'
        print(data_str)

        """Update lr value"""
        lr = scheduler.get_last_lr()


if __name__ == "__main__":
    total_train_procedure(load_model=False, starting_epoch=0, starting_lr=0.5)
