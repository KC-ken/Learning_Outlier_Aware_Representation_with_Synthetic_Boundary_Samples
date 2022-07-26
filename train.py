# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import argparse
import importlib
import time
import logging
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from yaml import parse
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter

from models import SupResNet, SSLResNet
import data
import trainers
from losses import SupConLoss
from vo_losses import VOConLoss
from utils import *
from fast_eval_ssd import fast_eval
import wandb
import random

def evaluate(val, model, device, test_loader, in_train_loader, in_test_loader, norm_layer, OODs, criterion, args, epoch):
    ## eval accuracy
    prec1, _ = val(model, device, test_loader, criterion, args, epoch)
    
    # compute OOD detection performance
    fpr95, auroc, aupr = fast_eval(
        model=model,
        train_loader=in_train_loader,
        test_loader=in_test_loader,
        OODs=OODs,
        adataset=args.dataset,
        atraining_mode=args.training_mode
    )

    return prec1, fpr95, auroc, aupr

def main():
    parser = argparse.ArgumentParser(description="SSD evaluation")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="/data/data_vvikash/fall20/SSD/trained_models/",
    )  # change this
    parser.add_argument("--exp-name", type=str, default="temp")
    parser.add_argument(
        "--training-mode", type=str, choices=("SimCLR", "SupCon", "SupCE")
    )

    # model
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--num-classes", type=int, default=10)

    # training
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-dir", type=str, default="./datasets/")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=10)

    # ssl
    parser.add_argument(
        "--method", type=str, default="SupCon", choices=["SupCon", "SimCLR", "SupCE"]
    )
    parser.add_argument("--temperature", type=float, default=0.5)

    # misc
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--ckpt", type=str, help="checkpoint path")
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--virtual-outlier", action="store_true", default=False)
    parser.add_argument("--lamb", type=float, default=1)
    parser.add_argument("--near-region", type=float, default=0.1)
    parser.add_argument("--default-warmup", action="store_true", default=False)
    parser.add_argument(
        "--vos-mode", type=str, default="Cont", choices=["Cont", "DualCont", "DualOut", "ContOne"]
    )
    parser.add_argument("--resample", action="store_true", default=False)
    parser.add_argument("--normalize-ID", action="store_true", default=False)
    parser.add_argument("--grad-head", action="store_true", default=False)

    args = parser.parse_args()
    device = "cuda:0"

    if args.batch_size > 256 and not args.warmup:
        warnings.warn("Use warmup training for larger batch-sizes > 256")

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # create resutls dir (for logs, checkpoints, etc.)
    result_main_dir = os.path.join(args.results_dir, args.exp_name)

    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = result_sub_dir = os.path.join(
            result_main_dir,
            "{}--dataset-{}-arch-{}-lr-{}_epochs-{}".format(
                n + 1, args.dataset, args.arch, args.lr, args.epochs
            ),
        )
    else:
        os.mkdir(result_main_dir)
        result_sub_dir = result_sub_dir = os.path.join(
            result_main_dir,
            "1--dataset-{}-arch-{}-lr-{}_epochs-{}".format(
                args.dataset, args.arch, args.lr, args.epochs
            ),
        )
    create_subdirs(result_sub_dir)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)

    # initial wandb
    wandb.init(project="SSD-training", name=args.exp_name)
    wandb.config = {
        "arch": args.arch,
        "training_method": args.method,
        "dataset": args.dataset,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Create model
    if args.training_mode in ["SimCLR", "SupCon"]:
        model = SSLResNet(arch=args.arch).to(device)
    elif args.training_mode == "SupCE":
        model = SupResNet(arch=args.arch, num_classes=args.num_classes).to(device)
    else:
        raise ValueError("training mode not supported")

    # load feature extractor on gpu
    model.encoder = torch.nn.DataParallel(model.encoder).to(device)

    # Dataloader for training
    train_loader, test_loader, _ = data.__dict__[args.dataset](
        args.data_dir,
        mode="ssl" if args.training_mode in ["SimCLR", "SupCon"] else "org",
        normalize=args.normalize,
        size=args.size,
        batch_size=args.batch_size,
    )

    # Dataloader for evaluating
    in_train_loader, in_test_loader, norm_layer = data.__dict__[args.dataset](
        args.data_dir,
        args.eval_batch_size,
        mode="base",
        normalize=args.normalize,
        size=args.size,
    )

    # ood dataset for fast_eval
    eval_ds = ["cifar10", "cifar100", "svhn", "texture", "blobs"]
    eval_ds.remove(args.dataset)

    # OOD loader
    OODs = ()
    for d in eval_ds:
        _, ood_loader, _ = data.__dict__[d](
                args.data_dir,
                args.eval_batch_size,
                mode="base",
                normalize=args.normalize,
                norm_layer=norm_layer,
                size=args.size,
            )

        OODs = ((d, ood_loader),) + OODs

    # criterion
    if args.virtual_outlier:
        print(f"@@@@using virtual outlier sampling with lambda {args.lamb}....")
        print(f"@@@@vos mode: {args.vos_mode}...")
        criterion = VOConLoss(temperature=args.temperature, lamb=args.lamb, vos_mode=args.vos_mode).cuda()
    else:
        # TODO rewrite this
        print(f"@@@using normal {args.training_mode}")
        criterion = VOConLoss(temperature=args.temperature).cuda()
        # criterion = (
        #     SupConLoss(temperature=args.temperature).cuda()
        #     if args.training_mode in ["SimCLR", "SupCon"]
        #     else nn.CrossEntropyLoss().cuda()
        # )

    if args.default_warmup:
        print("@@@@default warm up....")
        # criterion_warm = (
        #     SupConLoss(temperature=args.temperature).cuda()
        #     if args.training_mode in ["SimCLR", "SupCon"]
        #     else nn.CrossEntropyLoss().cuda()
        # )
        criterion_warm = criterion
    else:
        criterion_warm = criterion

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # select training and validation methods
    trainer = (
        trainers.ssl
        if args.training_mode in ["SimCLR", "SupCon"]
        else trainers.supervised
    )
    val = knn if args.training_mode in ["SimCLR", "SupCon"] else baseeval
    
    num_steps = 0

    # warmup
    if args.warmup:
        wamrup_epochs = args.warmup_epochs
        print(f"Warmup training for {wamrup_epochs} epochs")
        warmup_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.01,
            max_lr=args.lr,
            step_size_up=wamrup_epochs * len(train_loader),
        )
        for epoch in range(wamrup_epochs):
            trainer(
                model,
                device,
                train_loader,
                criterion_warm,
                optimizer,
                warmup_lr_scheduler,
                epoch,
                args,
                args.virtual_outlier & ~args.default_warmup,
                args.resample,
                args.near_region,
                args.normalize_ID,
                args.grad_head,
            )

            ## eval
            prec1, fpr95, auroc, aupr = evaluate(val, model, device, test_loader, in_train_loader, in_test_loader, norm_layer, OODs, criterion_warm, args, epoch)
            
            wandb.log({"validation accuracy": prec1}, step=num_steps)

            for i, (ds, _) in enumerate(OODs):
                wandb.log({f"{ds}_fpr95": fpr95[i], f"{ds}_auroc": auroc[i], f"{ds}_aupr": aupr[i]}, step=num_steps)
            
            num_steps += 1

    best_prec1 = 0

    for p in optimizer.param_groups:
        p["lr"] = args.lr
        p["initial_lr"] = args.lr
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(train_loader), 1e-4
    )

    for epoch in range(0, args.epochs):
        trainer(
            model, device, train_loader, criterion, optimizer, lr_scheduler, epoch, args, args.virtual_outlier, args.resample, args.near_region, args.normalize_ID, args.grad_head
        )

        prec1, fpr95, auroc, aupr = evaluate(val, model, device, test_loader, in_train_loader, in_test_loader, norm_layer, OODs, criterion, args, epoch)

        # remember best accuracy and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        d = {
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_prec1": best_prec1,
            "optimizer": optimizer.state_dict(),
        }

        save_checkpoint(
            d,
            is_best,
            os.path.join(result_sub_dir, "checkpoint"),
        )

        if not (epoch + 1) % args.save_freq:
            save_checkpoint(
                d,
                is_best,
                os.path.join(result_sub_dir, "checkpoint"),
                filename=f"checkpoint_{epoch+1}.pth.tar",
            )

        logger.info(
            f"Epoch {epoch}, validation accuracy {prec1}, best_prec {best_prec1}"
        )
        
        wandb.log({"validation accuracy": prec1}, step=num_steps)

        for i, (ds, _) in enumerate(OODs):
            wandb.log({f"{ds}_fpr95": fpr95[i], f"{ds}_auroc": auroc[i], f"{ds}_aupr": aupr[i]}, step=num_steps)
        # clone results to latest subdir (sync after every epoch)
        clone_results_to_latest_subdir(
            result_sub_dir, os.path.join(result_main_dir, "latest_exp")
        )

        num_steps += 1


if __name__ == "__main__":
    main()
