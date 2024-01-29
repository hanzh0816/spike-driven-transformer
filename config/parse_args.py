import argparse
import numpy as np


def parse_option() -> argparse.ArgumentParser:
    """
    add command line arguments parsing options
    """
    parser = argparse.ArgumentParser(description="SpikeDrivenTransformer Settings")
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--batch_size", type=int, help="batch size for single GPU")
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
