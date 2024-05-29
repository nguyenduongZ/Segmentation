from dataset import get_ds
from .model import Unet
from metrics import miou
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
import wandb
import hashlib
import json
import os

def get_hash(args):
    args_str = json.dumps(vars(args), sort_keys=True)
    args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
    return args_hash

def trainer(args):
  
    #set up device
    if torch.cuda.is_available():
        device = torch.device("cuda", index=args.idx)
    else:
        device = torch.device("cpu")

    args, train_dl, val_dl, test_dl = get_ds(args)

    print(f"#TRAIN batch: {len(train_dl)}")
    print(f"#VAL batch: {len(val_dl)}")
    print(f"#TEST batch: {len(test_dl)}")

    run_name = get_hash(args)

    if args.log:
        run = wandb.init(
        project='seg',
        entity='truelove',
        config=args,
        name=run_name,
        force=True
        )

    run_dir = os.getcwd() + '/runs'
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    sv_dir = run_dir + f"/{run_name}"
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)

    best_model_path = sv_dir + f'/best.pt'
    last_model_path = sv_dir + f'/last.pt'

    model = Unet(args).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, len(train_dl) * args.epochs)

    old_valid_loss = 1e26
    
    for epoch in range(args.epochs):
        log_dict = {}

        model.train()
        totol_loss = 0 
        total_iou = 0
        for _, (img, target) in enumerate(train_dl):
            img = img.to(device)
            target = target.to(device)

            pred = model(img)

            # loss = 
