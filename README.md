# 3D Menu Visualization

This project aims to produce a deep learning model capable of quickly providing high-quality, interactive 3D menu items.

## Usage

Is the layer of indirection worth it? Why not just store a list of commands...

1. Create an experiment configuration file. Example:

```yaml
# my_model_config.yml
model:
  name: gaussian_splatting
dataset:
  input: ./data/input/tandt/truck
  output: ./data/output/
logging:
  path: ./experiments/
training:
  iterations: 50
```

The configuration specifies which model to run, which input data, where to log, and training parameters.

2. Define your model code. The minimum required structure:

```text
models/
  gaussian_splatting/
    __init__.py
    model.py
    train.py
```

where `train.py` is the main entry point. It should accept command line arguments corresponding to the keys and values in the `training` key of the configuration file.

> For example, the `train.py` for `gaussian_splatting` should accept an argument called `iterations` because that is what is specificed in the configuration file.

3. Run the configuration

```bash
python train.py --config ./configs/my_model_config.yaml
```

4. Evaluate the model

```bash
python eval.py --config ./configs/my_model_config.yaml
```

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