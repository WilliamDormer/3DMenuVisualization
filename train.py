"""train

Main module for training models for 3D novel view synthesis.
"""
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import NamedTuple, Callable, Optional
import importlib
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import TrainingArgs

@dataclass
class Args:
    scene: Path
    config: dict[str, any]
    allow_cpu: bool
    disable_logs: bool

class ModelImports(NamedTuple):
    """Imported components from a model subpackage.
    
    Each model is defined as a subpackage found in the `models/` directory.
    This class defines the interface that the subpackage should expose to the main training module.
    """
    model: nn.Module
    # Accepts arguments, constructs argument parsers as needed, returns TrainingArgs
    setup_func: Callable[[list[str]], TrainingArgs]
    # Optimizer, Loss, Regularization, etc. are implementation details of the training function
    train_func: Callable[[TrainingArgs, SummaryWriter, Optional[DataLoader], str], None]
    data_loader: Callable[[Path], DataLoader]

def import_model(model_name: str) -> ModelImports:
    """Dynamically import ModelImports components from `model_name` subpackage"""

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

def extract_args(config: dict) -> list[str]:
    """Extracts and formats command line arguments from a configuration file"""
    args: list[str] = []
    for k, v in config["training"].items():
        k = f"--{k}"
        args.append(str(k))
        args.append(str(v))
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to model config file", required=True)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow training with CPU")
    args: Args = parser.parse_args()

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

    # Load model configuration file
    with open(args.config, 'r') as file:
        config: dict = yaml.safe_load(file)

    args = extract_args(config)

    # Import model code into parent context
    model_name = config["model"]["name"]
    model_interface = import_model(model_name)

    # Call setup
    training_args = model_interface.setup_func(args)

    data_loader = None
    try:
        data_loader = model_interface.data_loader(config["dataset"]["path"], training_args)
    except KeyError as err:
        print(err)

    # Construct writer
    # TODO - replace with our custom Logger class? Would need to modify each model files, not to much work
    writer = SummaryWriter(
        config["logging"]["path"],
    )

    # Instantiate training loop
    #   Pass training arguments from config file to the model's `train` module
    #   Pass writer
    model_interface.train_func(
        training_args,
        writer,
        data_loader,
        device,
    )

    writer.close()
