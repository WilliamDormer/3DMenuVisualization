# 3D Menu Visualization

This project aims to produce a deep learning model capable of quickly providing high-quality, interactive 3D menu items.

## TODO

- [ ] Move `video_to_images.py` scipt a `scripts/` folder; consider wrapping with `torch` DataLoader (https://pytorch.org/vision/stable/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder)
- [ ] Test 'one-click' environment setup in fresh folder on CSLab and Windows device
- [ ] Check that all models have appropriate experiment tracking/logging using wandb (or optionally Tensorboard?)
- [ ] Try creating a 'main entry point' `train.py` script that takes some configuration values and runs an end-to-end pipeline for a user-specified dataset and model
- [ ] Update installation docs to include a troubleshooting section, update architecture docs to include description of parameters

## Usage

...

## Folder Structure 

`data/`
this folder contains scripts to load and pre-process datasets. Use dataset_loader.py script to abstract the specifics of loading the different datasets.

`models/`
Each model should be defined in a separate file for clarity and modularity. You can include multiple models in this folder.

`configs/`
use a YAML file to store hyperparameters and other configurations. This makes it easy to tweak and manage different experiments

`experiments/`
for logging and saving results. Each experiment can have it's own folder where you store model checkpoints, logs, and plots. 
    checkpoints: save trained model weights for future evaluation
    logs: store training logs (e.g. loss, accuracy per epoch)

`utils/`
Utilities such as logging class to track metrics during training, or functions to compute error metrics or evaluation metrics.

`train.py`
This script orchestrates the training process. It loads models, datasets and hyperparameters from the config file, and tracks progress using a logging mechanism.

`evaluate.py`
This script can be used to evaluate a trained model on validation or test data.

`video_to_images.py`
FFmpeg script we wrote to enable converting custom videos into input images for the models.

`requirements.txt`
List all dependencies here, making the project easy to install and run on different environments. Refer to [https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html) for details.
