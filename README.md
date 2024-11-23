# 3D Menu Visualization

This project aims to produce a deep learning model capable of quickly providing high-quality, interactive 3D menu items.


```bash
scp -r /w/246/kappa/projects/3DMenuVisualization/data/output/gaussian_splatting/croissant_sample . 
```

## Workflow

1. Mickell prepares MetaFood3D data

  1. Select a directory of food items
  2. Run COLMAP to generate Structure-From-Motion points
  3. `scp` the results to our repo's `data/input/meta_food_3d` folder

2. We run our model on the input and generate the splatts

## Solution

Structure model files like so:

```text
models/
  <model>/
    __init__.py
    model.py
    train.py
```

And have a main `train.py` script that:

1. Takes config values, like training parameters, and feeds it to the `<model>/train.py` script
2. Passes a logger to the `<model>/train.py` script
3. Passes a dataloader to the `<model>/train.py` script

Essentially, in `train.py`

```python
# train.py

model = ... # import 'models/<model>'

training_args = model.setup(config)

data_loader = model.get_data_loader(config)
# Alternatively: data.get_data_loader(config)

logger = ... # e.g. tensorboard.SummaryWriter

model.train(
  training_args,
  data_loader,
  logger,
)
```

Note that specific training components are hidden away as implementation details because they are typically specific to a model.
E.g. the `model.train` function would have code for the loss function, optimizer, regularizers, etc. and is not dictated by the configuration file.















1. Configuration loader - each model can have unique set of arguments
2. Data loader - prepares the input data
3. Model
4. Optimizer & Loss
5. Logging - using the `torch.utils.tensorboard.SummaryWriter`
6. Training
7. Data Loader - save output
8. Evaluation - compute metrics, save results, update rankings


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
  <model>/
    __init__.py
    model.py
    train.py
```

`train.py` should have the following functions:

```python
# train.py
def setup():
  ...

def train():
  ...
```

where `train.py` is the main entry point. The values of config[`training`]. 

> For example, the `train.py` for `gaussian_splatting` should accept an argument called `--iterations` because that is what is specificed in the configuration file.

3. Run the configuration

```bash
python train.py --config ./configs/my_model_config.yaml
```

4. Evaluate the model

```bash
python eval.py --config ./configs/my_model_config.yaml
```

## Folder Structure


We need an easy way to train and evaluate many slightly different Novel-view synthesis models.

That is, the repo structure should enable us to:

- read input from a shared `data/` folder and pass it to models
- write model outputs to a shared `data/` folder
- log (with Tensorboard) any kind of model

- create models independently
- create & swap components of models (optimizer, loss, layers, blocks, components) independently

- train models independently
- train models that require varying counts and kinds of arguments

- evaluate models independently
- evaluate models using a predefined set of metrics
- produce rankings of model performance after evaluating a model

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

# Workflow

```
srun --partition=gpunodes -c 2 --mem=8G --gres=gpu:rtx_4090:1 -t 60 --pty bash --login
```

Detailed documentation for running with GPU:

https://pyrite-pigment-7b7.notion.site/Tutorial-Gaussian-Splatting-133697eeb9a580dab39fd80af9e6cdda