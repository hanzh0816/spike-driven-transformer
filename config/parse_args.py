import argparse
from .default import get_config


def parse_option():
    """
    add command line arguments parsing options
    """

    parser = argparse.ArgumentParser(description="SpikeDrivenTransformer Settings")

    # ----------------------------------------------------------------
    # misc settings
    # ----------------------------------------------------------------
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
    parser.add_argument("--tag", help="tag of experiment")

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
    parser.add_argument(
        "--output",
        default="output",
        required=True,
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
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

    args, _ = parser.parse_known_args()
    config = get_config(args)
    return args, config
