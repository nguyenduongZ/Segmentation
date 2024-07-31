import os, sys

from .unet import Unet


model_dict = {
    "unet" : {
        "oxford" : Unet
    },
    # "segnet" : {
    #     "oxford" : OxFordPetSegNet
    # }
}