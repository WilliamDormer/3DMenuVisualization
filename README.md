# 3D Menu Visualization

This project aims to produce a deep learning model capable of quickly providing high-quality, interactive 3D menu items.

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

`requirements.txt`
List all dependencies here, making the project easy to install and run on different environments. Refer to [https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html) for details.