"""benchmark

Script for benchmarking models on a specific scene
"""

import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=Path)
    args = parser.parse_args()

    # Train & evaluate each model in the `models` folder on `--scene`
