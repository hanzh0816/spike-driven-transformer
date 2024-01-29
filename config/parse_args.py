import argparse
import numpy as np


def parse_option() -> argparse.ArgumentParser:
    """
    add command line arguments parsing options
    """
    parser = argparse.ArgumentParser(description="SpikeDrivenTransformer Settings")
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument
