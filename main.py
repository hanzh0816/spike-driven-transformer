import os
import argparse
import yaml
import logging
import numpy as np
from accelerate import Accelerator
import torch
from timm import create_model
from datasets import build_loader


def main(accelerator: Accelerator, config, logger):
    data_loader_train, data_loader_test = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    processor, model = build_model(config)

def train_one_epoch():
    pass


@torch.no_grad()
def validate():
    pass


if __name__ == "__main__":
    accelerator = Accelerator()

    main()
