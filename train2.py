"""train2

Entry point for training
"""
import torch
import argparse
import yaml
from pathlib import Path
import subprocess

if __name__ == "__main__":
    # Command line arguments for config file path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file", required=True)
    args = parser.parse_args()

    # https://www.geeksforgeeks.org/how-to-use-gpu-acceleration-in-pytorch/
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
        print("torch.version.cuda: ", torch.version.cuda)
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU. This is not ideal for training, so please check torch version!")
        # raise Exception("using wrong device (cpu instead of GPU)")


    with open(args.config, 'r') as file:
        config: dict = yaml.safe_load(file)

    # Find model in `models` directory
    model_name = config["model"]["name"]
    model_dir = Path(f"./models/{model_name}")

    args = []
    for k, v in config["training"].items():
        k = f"--{k}"
        args.append(k)
        args.append(v)

    print(model_dir, args)

    # TODO - make cross-platform: `python3` vs `python`
    cmd = ["python3", model_dir/"train.py", *args]
    print(cmd)
    subprocess.run(cmd)