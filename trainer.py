from dataset import get_ds
from model import Unet
from metrics import miou
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

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
    # print(torch_mask.shape())
    #set up device
    if torch.cuda.is_available():
        device = torch.device("cuda", index=args.idx)
    else:
        device = torch.device("cpu")

    args, train_dl, val_dl, test_dl = get_ds(args)

    print(f"#TRAIN batch: {len(train_dl)}")
    print(f"#VAL batch: {len(val_dl)}")
    print(f"#TEST batch: {len(test_dl)}")

    # run_name = get_hash(args)
    now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    if args.log:
        run = wandb.init(
        project='seg',
        # entity='truelove',
        config=args,
        name=now,
        force=True
        )

    run_dir = os.getcwd() + '/runs'
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    sv_dir = run_dir + f"/{now}"
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)

    best_model_path = sv_dir + f'/best.pt'
    last_model_path = sv_dir + f'/last.pt'

    model = Unet(args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params}")
    total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_train_params}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, len(train_dl) * args.epochs)

    old_valid_loss = 1e26
    
    for epoch in range(args.epochs):
        log_dict = {}

        model.train()
        total_loss = 0 
        total_iou = 0
        for _, (img, target) in enumerate(train_dl):
            img = img.to(device)
            target = target.to(device)

            pred = model(img)
            loss = nn.CrossEntropyLoss()(pred, target)
            iou = miou(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            total_iou += iou

        train_mean_loss = total_loss / len(train_dl)
        train_iou = total_iou / len(train_dl)

        log_dict['train/loss'] = train_mean_loss 
        log_dict['train/iou'] = train_iou 

        print(f"Epoch: {epoch} - Train loss: {train_mean_loss} - Train mIOU: {train_iou}")

        model.eval()    
        with torch.no_grad():
            total_loss = 0
            total_iou = 0
            for _, (img, target) in enumerate(val_dl):
                img = img.to(device)
                target = target.to(device)

                pred = model(img)
                loss = nn.CrossEntropyLoss()(pred, target)
                iou = miou(pred, target)

                total_loss += loss.item()
                total_iou += iou
        valid_mean_loss = total_loss / len(val_dl)
        valid_iou = total_iou / len(val_dl)

        log_dict['valid/loss'] = valid_mean_loss
        log_dict['valid/miou'] = valid_iou

        print(f"Epoch: {epoch} - Valid loss: {valid_mean_loss} - Valid mIOU: {valid_iou}")

        save_dict = {
            'args' : args,
            'model_state_dict' : model.state_dict()
        }

        if valid_mean_loss < old_valid_loss:
            old_valid_loss = valid_mean_loss

            torch.save(save_dict, best_model_path)
        torch.save(save_dict, last_model_path)  

        if args.log:
            run.log(log_dict)

    if args.log:
        run.log_model(path=best_model_path, name=f'{now}-best-model') 
        run.log_model(path=last_model_path, name=f'{now}-last-model')