import os
import time
import datetime
from tqdm import tqdm
import torch.nn.functional as F

import torch
from accelerate import Accelerator
import torch.nn as nn
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import accuracy, AverageMeter

from datasets import build_loader
from model import build_model
from config import parse_option
from utils import set_logger, init_seed


def test(args, config, logger):
    _, _, data_loader_train, data_loader_val = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.NAME}")
    model = build_model(config)
    model.to(device)
    # use timm.optim to create specific optimizer
    optimizer = create_optimizer(args=args, model=model)

    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()

    t = tqdm(data_loader_train)
    t.set_description("Processing:")

    for idx, (inputs, labels) in enumerate(t):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)[0]
        loss = criterion(outputs, labels)

        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        break
        # loss = criterion(outputs, labels.long())
        # loss.backward(retain_graph=True)
        # optimizer.step()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    device = "cuda:4"
    args, config = parse_option()
    logger = set_logger(config=config)
    init_seed(config)

    test(args, config, logger)

    # main(accelerator, args, config, logger)
