# build lr_scheduler
import os
import numpy as np
from sympy import Q
import torch
import random
import torch.nn as nn
from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
    BinaryCrossEntropy,
)

from yacs.config import CfgNode as CN
import wandb


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)


def init_seed(config):
    seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_loss_fn(config):
    if config.TRAIN.LABEL_SMOOTH != 0:
        if config.TRAIN.BCE_LOSS:
            train_loss_fn = BinaryCrossEntropy(smoothing=config.TRAIN.LABEL_SMOOTH)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(
                smoothing=config.TRAIN.LABEL_SMOOTH
            )
    else:
        if config.TRAIN.BCE_LOSS:
            train_loss_fn = BinaryCrossEntropy()
        else:
            train_loss_fn = nn.CrossEntropyLoss()

    return train_loss_fn


def get_optimizer_args(config):
    opt_dict = {
        "opt": config.OPTIMIZER.OPT,
        "lr": config.LR_SCHEDULER.LR,
        "weight_decay": config.OPTIMIZER.WEIGHT_DECAY,
        "momentum": config.OPTIMIZER.MOMENTUM,
        "opt_eps": config.OPTIMIZER.EPS,
        "opt_betas": config.OPTIMIZER.BETAS,
    }
    cfg = DictToObject(opt_dict)
    return cfg


def get_lr_scheduler_args(config):
    sched_dict = {
        "eval_metric": config.EVAL_METRIC,
        "sched": config.LR_SCHEDULER.SCHED,
        "epochs": config.TRAIN.EPOCHS,
        "decay_epochs": config.LR_SCHEDULER.DECAY_EPOCHS,
        "warmup_epochs": config.LR_SCHEDULER.WARMUP_EPOCHS,
        "cooldown_epochs": config.LR_SCHEDULER.COOLDOWN_EPOCHS,
        "warmup_lr": config.LR_SCHEDULER.WARMUP_LR,
        "min_lr": config.LR_SCHEDULER.MIN_LR,
    }
    cfg = DictToObject(sched_dict)
    return cfg


def wandb_init(config, device):

    wandb.init(
        project="spike-driven transformer",
        config=config,
        entity="snn-training",
        job_type="training",
        reinit=True,
        dir=config.OUTPUT,
        tags=config.TAG,
        name=config.TAG + "process" + str(device),
    )


def save_model(accelerator, config, epoch, model_without_ddp, optimizer):
    epoch_name = str(epoch)
    output = os.path.join(config.OUTPUT, ("checkpoint-%s.pth" % epoch_name))
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if accelerator.is_main_process:
        torch.save(to_save, output)
        print(f"Saved checkpoint-{epoch_name}.pth")


def load_model(config, model_without_ddp, optimizer):
    checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
    print(f"Resume checkpoint {config.MODEL.RESUME}")
