import argparse
import numpy as np
from .default import get_config


def parse_option():
    """
    add command line arguments parsing options
    """

    parser = argparse.ArgumentParser(description="SpikeDrivenTransformer Settings")
    parser.add_argument(
        "--cfg", type=str, required=True, metavar="FILE", help="path to config file"
    )
    parser.add_argument("--eval", type=bool, help="evaluate mode")
    parser.add_argument(
        "--eval-metric",
        default="top1",
        type=str,
        metavar="EVAL_METRIC",
        help='Best metric (default: "top1")',
    )

    # ----------------------------------------------------------------
    # data settings
    # ----------------------------------------------------------------

    parser.add_argument("--batch_size", type=int, help="batch size for single GPU")
    parser.add_argument("--vb_size", type=int, help="batch size for val")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        metavar="DATASET",
        help="dataset name to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        metavar="DATA_PATH",
        help="path to dataset",
    )
    parser.add_argument("--resume", type=bool, help="resume from checkpoint")
    parser.add_argument("--resume_path", type=str, help="resume path ")
    parser.add_argument("--init_checkpoint", type=bool, help="init checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="init checkpoint path")
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

    # ----------------------------------------------------------------
    # train settings
    # ----------------------------------------------------------------

    parser.add_argument("--start_epoch", type=int, default=0, help="start epoch")
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )

    # ----------------------------------------------------------------
    # optimizer settings
    # ----------------------------------------------------------------

    parser.add_argument(
        "--opt", type=str, default="sgd", metavar="OPTIMIZER", help="optimizer"
    )
    parser.add_argument(
        "--opt-eps",
        default=None,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: None, use opt default)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Optimizer momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="weight decay (default: 0.0001)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--clip-mode",
        type=str,
        default="norm",
        help='Gradient clipping mode. One of ("norm", "value", "agc")',
    )

    # ----------------------------------------------------------------
    # lr scheduler settings
    # ----------------------------------------------------------------
    parser.add_argument(
        "--sched",
        default="step",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "step"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="warmup learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config
