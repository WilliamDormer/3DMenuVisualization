"""null_model.train"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Your name", required=True)
    args = parser.parse_args()

    print("Null model w/ command line argument")
    print(f"Hello, {args.name}!")