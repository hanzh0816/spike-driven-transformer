import os
import yaml
import logging
import torch.distributed as dist
from yacs.config import CfgNode as CN

_C = CN()

# ----------------------------------------------------------
# basic settings
# ----------------------------------------------------------
_C.BASE = [""]
_C.LOG_LEVEL = logging.INFO

