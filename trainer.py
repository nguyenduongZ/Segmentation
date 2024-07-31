import os, sys
from rich.progress import track
import random
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

from dataset import get_ds
from model import get_model, model_dict, Unet
from metrics import miou
from torchmetrics.classification import Accuracy
import wandb


def trainer(args):
    
    # seed setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    #set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=args.idx)

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

    class Model(model_dict[args.model][args.ds]):
        def __init__(self, args):
            super().__init__(args)

            self.device = device

    model = Model(args).to(device)
    # model = Unet(args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params}")
    total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_train_params}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, len(train_dl) * args.epochs)

    task = 'multiclass' if args.clf_n_classes != 1 else 'binary'
    clf_accuracy = Accuracy(task=task, num_classes=args.clf_n_classes).to(device)
    old_valid_loss = 1e26
    
    for epoch in range(args.epochs):
        log_dict = {}

        model.train()
        total_loss = 0 
        total_iou = 0
        total_clf_acc = 0

        for _, (img, target) in enumerate(train_dl):
            img = img.to(device)
            for task_key in target:
                target[task_key] = target[task_key].to(device)
            # target = target.to(device)

            pred = model(img)
            clf_target = target['category']
            seg_target = target['semantic']

            clf_loss = nn.CrossEntropyLoss()(pred["category"], clf_target)
            seg_loss = nn.CrossEntropyLoss()(pred["semantic"], seg_target)

            loss = clf_loss + seg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            clf_acc = clf_accuracy(pred["category"], clf_target)
            iou = miou(pred["semantic"], seg_target)

            total_loss += loss.item()
            total_iou += iou
            total_clf_acc += clf_acc.item()

        train_mean_loss = total_loss / len(train_dl)
        train_iou = total_iou / len(train_dl)
        train_clf_acc = total_clf_acc / len(train_dl)

        log_dict['train/loss'] = train_mean_loss 
        log_dict['train/iou'] = train_iou 
        log_dict['train/clf_acc'] = train_clf_acc

        print(f"Epoch: {epoch} - Train loss: {train_mean_loss} - Train mIOU: {train_iou} - Train Clf: {train_clf_acc}")

        model.eval()    
        with torch.no_grad():
            total_loss = 0
            total_iou = 0
            for _, (img, target) in enumerate(val_dl):
                img = img.to(device)
                for task in target:
                    target[task] = target[task].to(device)
                # target = target.to(device)

                pred = model(img)
                clf_target = target['category']
                seg_target = target['semantic']

                clf_loss = nn.CrossEntropyLoss()(pred["category"], clf_target)
                seg_loss = nn.CrossEntropyLoss()(pred["semantic"], seg_target)

                loss = clf_loss + seg_loss

                clf_acc = clf_accuracy(pred["category"], clf_target)
                iou = miou(pred["semantic"], seg_target)

                total_loss += loss.item()
                total_iou += iou
                total_clf_acc += clf_acc.item()

        valid_mean_loss = total_loss / len(val_dl)
        valid_iou = total_iou / len(val_dl)
        valid_clf_acc = total_clf_acc / len(val_dl)

        log_dict['valid/loss'] = valid_mean_loss
        log_dict['valid/miou'] = valid_iou
        log_dict['valid/clf_acc'] = valid_clf_acc

        print(f"Epoch: {epoch} - Valid loss: {valid_mean_loss} - Valid mIOU: {valid_iou} - Valid Clf: {valid_clf_acc}")

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