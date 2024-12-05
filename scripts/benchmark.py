"""benchmark.py

Module for running benchmarks
"""

import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="Path to config")
    args = parser.parse_args()

    config = tomllib.load(args.config)
    print(config)
