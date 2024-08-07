from trainer import trainer

import argparse
import numpy as np
import torch
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Segmentation')

    # DATASET
    parser.add_argument('--ds', type=str, required=True, choices=['oxford'], help='dataset used in training')

    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--wk', type=str, default=1, help='number of workers')
    parser.add_argument('--pm', action='store_true', help='pin memory')
    parser.add_argument('--sz', type=int, default=256, help='size of processed image')
    parser.add_argument('--aug', action='store_true', help='augmentation')

    # TRAINING
    parser.add_argument('--idx', type=int, default=0, help='device index used in training')
    parser.add_argument('--seed', type=int, default=0, help='seed used in training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs used in training')

    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'segnet'], help='backbone used in training')
    parser.add_argument('--loss', type=str, default='', choices=[], help='loss function used in training')

    parser.add_argument('--test', action='store_true', help='toggle to say that this experiment is just flow testing')

    # LOGGING
    parser.add_argument('--wandb', action='store_true', help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true', help='wandb logging')

    # MODEL
    parser.add_argument('--init_ch', type=int, default=32, help='number of kernel in the first')
    parser.add_argument('--clf_n_classes', type=int, default=37, help='channels of output')
    parser.add_argument('--seg_n_classes', type=int, default=3, help='channels of output')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    args = parser.parse_args()

    trainer(args)