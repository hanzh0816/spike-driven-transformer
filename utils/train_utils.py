# build lr_scheduler

from os import write
import numpy as np
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


def init_seed(config):
    seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_loss_fn(config):
    if config.TRAIN.LABEL_SMOOTH:
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


def wandb_init(config):
    wandb.init(
        project="spike-driven transformer",
        config=config,
        entity="snn-training",
        job_type="training",
        reinit=True,
        dir=config.OUTPUT,
        tags=config.EXPERIMENT,
    )
