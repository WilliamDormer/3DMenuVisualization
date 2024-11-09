import torch
import yaml
import argparse
import importlib
from pathlib import Path
from data.dataset_loader import get_dataloader
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.logger import Logger
from torch.utils.tensorboard import SummaryWriter
from models.garment_classifier.model import GarmentClassifier
from argparse import ArgumentParser
from utils import TrainingArgs
from .model import GarmentClassifier # Need relative import
import torchvision.transforms as transforms
import torchvision

def setup(parent_args: list[str]) -> TrainingArgs:
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    return TrainingArgs(parser.parse_args(parent_args))

def get_data_loader(data_path: Path, training: TrainingArgs) -> DataLoader:
    """The parent importer sets the data_path"""
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transform, download=True)
    return DataLoader(dataset, batch_size=training.args.batch_size, shuffle=True)

# example: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train(training: TrainingArgs, writer: SummaryWriter, dataloader: DataLoader, device: str) -> None:
    epochs: int = training.args.epochs
    learning_rate: float = training.args.learning_rate

    model = GarmentClassifier()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0
        last_loss = 0
        for i, data in enumerate(dataloader):
            # parse the input data and label information.
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero your gradients for every batch.
            optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(inputs)
            # compute the loss and its gradients
            loss = criterion(outputs, labels)
            loss.backward()
            # adjust learning weights
            optimizer.step()

            # Gather running data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch over the last 1000 batches
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch * len(dataloader) + i + 1
                writer.add_scalar('batch loss/train', last_loss, tb_x)
                running_loss = 0.
        
        writer.add_scalar('epoch last loss/train', loss.item(), epoch)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

        #TODO compute the validation loss as well, use it for early stopping. 
    writer.flush()


def get_model(model_class_name: str) -> nn.Module:
    """Dynamically import the Pytorch model `model_class_name` from the file `model_class_name.py`"""
    mod = importlib.import_module(f"models.{model_class_name.lower()}")
    cls = getattr(mod, model_class_name)
    return cls

def main() -> None:
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


    # --- Model, dataloader, optimizer, criterion

    # Load config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # -- Initialize Model
    model = GarmentClassifier()
    print(f"Training model {model}")

    # TODO modify this to select dataloader based on config file
    # TODO modify this to select optimizer based on config file
    # TODO modify this to select criterion based on config file

    # send the model to the device
    model = model.to(device)

    # model = Model1(config['model']['input_dim'], config['model']['output_dim'])
    dataloader = get_dataloader(dataset_identifier=None, data_path=config['dataset']['path'], batch_size=config['training']['batch_size'])
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Logging
    logger = Logger(log_dir=f"experiments/", config_path=args.config)
    
    # Train
    train(model, dataloader, criterion, optimizer, config['training']['epochs'], logger, device)

    # save the model parameters:
    final_parameter_save_path = logger.get_parameter_save_path()
    torch.save(model.state_dict(), final_parameter_save_path)

    logger.close()

    # TODO enforce that the user has all the requirements in requirements.txt
    # TODO add support for loading model parameters, for either continued training in the case of a break, or for evaluation.


if __name__ == "__main__":
    main()