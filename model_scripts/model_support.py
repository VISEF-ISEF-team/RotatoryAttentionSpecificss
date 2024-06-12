from networks.original_unet_attention import Attention_Unet
from networks.rotatory_attention_unet import Rotatory_Attention_Unet
from networks.rotatory_attention_unet_v2 import Rotatory_Attention_Unet_v2
from networks.rotatory_attention_unet_v3 import Rotatory_Attention_Unet_v3
from networks.unet import UNet
from networks.resunet import ResUNet
from networks.rotatory_resunet_attention import Rotatory_ResUNet_Attention
from networks.transunet import TransUNet
from networks.rotatory_transunet_attention import Rotatory_TransUNet_Attention


def get_models(model_name, color_channel, window_size, num_classes=8, image_size=128):
    # "unet", "rotatory_unet_attention", "rotatory_unet_attention_v3", "vit", "unetmer"
    if model_name == "unet_attention":
        model = Attention_Unet(inc=color_channel, outc=num_classes)

    elif model_name == "unet":
        model = UNet(inc=1, outc=num_classes)

    elif model_name == "resunet":
        model = ResUNet(inc=1, outc=num_classes)

    elif model_name == "rotatory_unet_attention":
        model = Rotatory_Attention_Unet(
            inc=color_channel, outc=num_classes, image_size=image_size, window_size=window_size)

    elif model_name == "rotatory_unet_attention_v2":
        model = Rotatory_Attention_Unet_v2(
            inc=color_channel, outc=num_classes, image_size=image_size, window_size=window_size)

    elif model_name == "rotatory_unet_attention_v3":
        model = Rotatory_Attention_Unet_v3(
            inc=color_channel, outc=num_classes, image_size=image_size, window_size=window_size)

    elif model_name == "rotatory_resunet_attention":
        model = Rotatory_ResUNet_Attention(
            inc=color_channel, outc=num_classes, image_size=image_size, window_size=window_size)

    elif model_name == "transunet":
        model = TransUNet(inc=color_channel, outc=num_classes,
                          image_size=image_size)

    elif model_name == "rotatory_transunet_attention":
        model = Rotatory_TransUNet_Attention(
            inc=color_channel, outc=num_classes, image_size=image_size, window_size=window_size)

    return model
