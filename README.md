# 3D Menu Visualization

This project aims to compare Gaussian Splatting models capable of quickly providing high-quality, interactive renders of 3D menu items.

## Contributions

In this work, we contribute:

1. Benchmarks of four Gaussian Splatting models on a subset of scenes from the [MetaFood3D dataset](http://arxiv.org/abs/2409.01966). Refer to [Benchmarks](./docs/BENCHMARKS.md) for details.
2. Bug fixes and new features for a few of the Gaussian Splatting models (such as the capability to export renders to video)
3. [DormerFood](/), a custom dataset to test realistic edge cases in 3D food rendering
3. FFmpeg script to enable converting custom videos into input images for the models. ([`video_to_images.py`](./scripts/video_to_images.py))
4. Documentation for model parameters

We compared the following models:

1. [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
2. [Self-Organizing Gaussian Grids](https://github.com/fraunhoferhhi/Self-Organizing-Gaussians/tree/main)
3. [GaussianShader](https://github.com/Asparagus15/GaussianShader)
4. [Few-Shot Gaussian Splatting](https://github.com/VITA-Group/FSGS)

## Install

Ensure you run the installation commands on a device with a CUDA-enabled GPU.
This installation guide assumes you are working on a Linux system.

First, clone the repo:

```bash
git clone https://github.com/WilliamDormer/3DMenuVisualization.git
cd 3DMenuVisualization
```

Next, make sure to set the appropriate environment variables for CUDA:

```bash
CUDA_HOME=/usr/local/cuda
TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
```

> You may need to use different values for `TORCH_CUDA_ARCH_LIST` depending on your system: [CUDA GPUs: Compute Capability (NVIDIA)](https://developer.nvidia.com/cuda-gpus)

Finally, create and activate the conda environment.

```bash
conda create --prefix ./env -f environment.yml
```

> Note that this `environment.yml` contains updated dependencies compared to the original implementations. The environment files supplied by the original papers were insufficient to replicate their environments.

### Troubleshooting

If you run into errors with setting up the environment or running the models, refer to the [Troubleshooting guide](./docs/TROUBLESHOOTING.md)

### Additional Required Dependencies

We recommend downloading system packages or precompiled binaries where possible.

> If you are on a system with restricted permissions (such as a university computing lab), then it is still possible to download/build external programs from source. Get in touch with your system administration for help.

All models:

- [COLMAP](https://colmap.github.io/install.html) - however note that FSGS uses a dockerized version of COLMAP with run with specific settings. Refer to the [FSGS Readme](https://github.com/VITA-Group/FSGS) for details.

GaussianShader

- [Nvdiffrast](https://nvlabs.github.io/nvdiffrast/)

Few-Shot Gaussian Splatting

- [LLFF](https://github.com/Fyusion/LLFF)

## Usage

1. Prepare you input data accordingly. Assuming you obtain a video (`.mp4`) of your scene, run the [`scripts/video_to_images.py`](./scripts/video_to_images.py) and save the resulting images to your `data/` folder
2. `cd` to your model of interest and run the `<model>/train.py` script. Refer to the model READMEs for details of how to pass arguments to the model
3. Note that the models are setup to log with [Weights & Biases](https://wandb.ai/site/). Create an account if needed. 

> (Or should we just stick to tensorboard?)

## Parameters

For a list of important parameters and parameter differences between models, refer to the [Parameters guide](./docs/PARAMS.md).

## Folder Structure

`data/`
this folder contains scripts to load and pre-process datasets. Use dataset_loader.py script to abstract the specifics of loading the different datasets.

`docs/`
Contains documentation related to benchmarks, troubleshooting, model parameters, etc.

`models/`
Each model should be defined in a separate file for clarity and modularity. You can include multiple models in this folder.

<!-- `configs/`
use a YAML file to store hyperparameters and other configurations. This makes it easy to tweak and manage different experiments -->

> TODO: Remove experiments? Or try running things with WANDB or Tensorboard?
`experiments/`
for logging and saving results. Each experiment can have it's own folder where you store model checkpoints, logs, and plots. 
    checkpoints: save trained model weights for future evaluation
    logs: store training logs (e.g. loss, accuracy per epoch)

`scripts/`
Scripts for data preprocessing.

<!-- `train.py`
This script orchestrates the training process. It loads models, datasets and hyperparameters from the config file, and tracks progress using a logging mechanism. -->

<!-- `evaluate.py`
This script can be used to evaluate a trained model on validation or test data. -->

`environment.yml`
List all dependencies here, making the project easy to install and run on different environments.
