from argparse import ArgumentParser, Namespace
import train


def main():
    """Setup arguments"""
    parser = ArgumentParser(description="Enter arguments for training")
    parser.add_argument("-r", "--rot", action="store_true", required=True)

    # model
    parser.add_argument("-m", "--model", choices=[
                        "unet_attention", "unet", "rotatory_unet_attention", "rotatory_unet_attention_v2",  "rotatory_unet_attention_v3", "vit", "rotatory_vit", "unetmer"], default="rotatory_unet_attention_v3", required=True)

    # loss
    parser.add_argument(
        "-l", "--loss", choices=["dice", "multi_dice", "tversky", "focal"], default="dice", required=True)

    # optimizer
    parser.add_argument(
        "--optim", choices=["adam", "adamw"], default="adam", required=True)

    # dataset name
    parser.add_argument(
        "--dataset", choices=["MMWHS", "CHD", "VHSCDD"], required=True, default="MMWHS")

    # training related param
    parser.add_argument("--epochs", type=int, default=25, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--classes", type=int, default=8, required=True)
    parser.add_argument("--size", type=int, default=256, required=True)
    parser.add_argument(
        "--precision", choices=["mixed", "normal"], default="normal")
    parser.add_argument("--startepoch", type=int, default=0)
    parser.add_argument("--startlr", type=float, default=0.01)
    parser.add_argument("--load", action="store_true")

    # transformer specific args

    args = parser.parse_args()

    """Process arguments"""
    rot = args.rot
    model_name = args.model
    loss_fn_name = args.loss
    optimizer_name = args.optim
    dataset_name = args.dataset
    num_epochs = args.epochs
    num_workers = args.workers
    batch_size = args.batch
    num_classes = args.classes
    image_size = args.size
    mixed_precision = args.precision
    starting_epoch = args.startepoch
    starting_lr = args.startlr
    load_model = args.load

    print(rot)
    return

    if mixed_precision == "normal":
        mixed_precision_boolean = False
    else:
        mixed_precision_boolean = True

    dataset_information = get_dataset(dataset_name=dataset_name)

    if rot:
        batch_size = None
        train.total_train_procedure(model_name=model_name, dataset_information=dataset_information,
                                    loss_fn_name=loss_fn_name, num_epochs=num_epochs, batch_size=batch_size, optimizer_name=optimizer_name, num_classes=num_classes, image_size=image_size, load_model=load_model, starting_epoch=starting_epoch,
                                    starting_lr=starting_lr
                                    )


if __name__ == "__main__":
    main()
