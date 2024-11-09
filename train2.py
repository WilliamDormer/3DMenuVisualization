"""train2

Main module for training competing models for 3D novel view synthesis.

Use this module to select:

1. a scene (input data)
2. a model configuration file

You may optionally:

- disable logging
- allow CPU
"""

import torch
import argparse
import yaml
from pathlib import Path
import subprocess
from dataclasses import dataclass
from typing import NamedTuple, Callable
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import importlib


from utils import TrainingArgs

@dataclass
class Args:
    scene: Path
    config: dict[str, any]
    allow_cpu: bool
    disable_logs: bool

class ModelImports(NamedTuple):
    """Imported components from a model subpackage.
    
    Each model is a subpackage found in the `models/` directory.
    This class defines the interface for imports from the subpackage.
    """
    model: nn.Module
    # Accepts arguments, constructs argument parsers as needed, returns TrainingArgs
    setup_func: Callable[[list[str]], TrainingArgs]
    # Optimizer, Loss, Regularization, etc. are implementation details of the training function
    # ArgStr is a string of arguments for parser.parse_args()
    train_func: Callable[[TrainingArgs, SummaryWriter], None]
    data_loader: Callable[[Path], DataLoader]


def import_model(model_name: str) -> ModelImports:
    """Dynamically import functions and entities from `model_name` subpackage"""

    module_name = f"models.{model_name.lower()}"
    model_class_name = "".join(map(str.title, model_name.split("_")))

    model_module = importlib.import_module(f"{module_name}.model")
    train_module = importlib.import_module(f"{module_name}.train")

    mi = ModelImports(
        model=getattr(model_module, model_class_name),
        setup_func=getattr(train_module, "setup"),
        train_func=getattr(train_module, "train"),
        data_loader=getattr(train_module, "get_data_loader"),
    )
    return mi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to model config file", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--disable-logs", action="store_true")
    args: Args = parser.parse_args()

    # https://www.geeksforgeeks.org/how-to-use-gpu-acceleration-in-pytorch/
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA (ver. {torch.version.cuda}) is available. Using GPU.")
    else:
        device = torch.device("cpu")
        if not args.allow_cpu:
            raise Exception("Using wrong device CPU instead of GPU")
        else:
            print("CUDA is not available. Using CPU.")
            print("This is not ideal for training, so please check torch version!")

    # Load model configuration
    with open(args.config, 'r') as file:
        config: dict = yaml.safe_load(file)

    # Find model in `models` directory
    model_name = config["model"]["name"]
    model_dir = Path(f"./models/{model_name}")
    if not model_dir.exists():
        raise ValueError(f"Could not find '{model_dir}'")

    # TODO: Is there some way to avoid stringifying the yaml key-values?
    args = []
    for k, v in config["training"].items():
        k = f"--{k}"
        args.append(str(k))
        args.append(str(v))

    print(model_dir, args)

    # Import model code into parent context
    mi = import_model(model_name)

    # Call setup
    training_args = mi.setup_func(args)
    print(training_args)

    # Pass scene path to dataset loader
    #   Impl details left to model designer
    data_loader = mi.data_loader(config["dataset"]["path"], training_args)
    print(data_loader)

    # Construct writer
    #   Configure output dir
    #   Configure comments
    writer = SummaryWriter(
        config["logging"]["path"],
        comment=f"LR_{config['training']['learning_rate']}_BATCH_{config['training']['batch_size']}"
    )

    # Instantiate training loop
    #   Pass training arguments from config file to the model's `train` module
    #   Pass writer
    #   Impl details left to model designer
    mi.train_func(training_args, writer, data_loader, device)


    writer.close()
