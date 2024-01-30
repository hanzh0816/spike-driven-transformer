import argparse
import numpy as np
from default import get_config


def parse_option():
    """
    add command line arguments parsing options
    """

    parser = argparse.ArgumentParser(description="SpikeDrivenTransformer Settings")
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")

    # data settings
    parser.add_argument("--batch_size", type=int, help="batch size for single GPU")
    parser.add_argument("--dataset", type=str, required=True, metavar="DATASET", help="dataset name to use")
    parser.add_argument("--data_path", type=str, required=True, metavar="DATA_PATH", help="path to dataset")
    parser.add_argument("--resume", type=bool, help="resume from checkpoint")
    parser.add_argument("--resume_path", type=str, help="resume path ")
    parser.add_argument(
        "--output",
        default="output",
        required=True,
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--save_freq", type=int, help="model save frequency")
    parser.add_argument("--print_freq", type=int, help="logger print frequency")

    # train settings
    parser.add_argument("--eval", type=bool, help="evaluat mode")
    parser.add_argument(
        "--opt", default="sgd", type=str, metavar="OPTIMIZER", help='Optimizer (default: "sgd")'
    )
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="Optimizer momentum (default: 0.9)"
    )
    parser.add_argument(
        "--sched", default="step", type=str, metavar="SCHEDULER", help='LR scheduler (default: "step")'
    )

    parser.add_argument(
        "--epochs", type=int, default=200, metavar="N", help="number of epochs to train (default: 200)"
    )
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config
