from networks.original_unet_attention import Attention_Unet
from networks.rotatory_attention_unet import Rotatory_Attention_Unet
from networks.rotatory_attention_unet_v2 import Rotatory_Attention_Unet_v2
from networks.rotatory_attention_unet_v3 import Rotatory_Attention_Unet_v3
from networks.unet import Unet
from networks.resunet import ResUNet


def get_models(model_name, num_classes=8, image_size=128):
    # "unet", "rotatory_unet_attention", "rotatory_unet_attention_v3", "vit", "unetmer"
    if model_name == "unet_attention":
        model = Attention_Unet(num_classes=num_classes)
    elif model_name == "unet":
        model = Unet(inc=1, outc=num_classes)
    elif model_name == "resunet":
        model = ResUNet(inc=1, outc=num_classes)
    elif model_name == "rotatory_unet_attention":
        model = Rotatory_Attention_Unet(image_size=image_size)
    elif model_name == "rotatory_unet_attention_v2":
        model = Rotatory_Attention_Unet_v2(image_size=image_size)
    elif model_name == "rotatory_unet_attention_v3":
        model = Rotatory_Attention_Unet_v3(image_size=image_size)

    return model
