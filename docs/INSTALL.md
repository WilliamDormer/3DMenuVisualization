# Install

Installation instructions for 3DMenuVisualization repository.

Ensure you run the installation commands on a device with a CUDA-enabled GPU.
This installation guide assumes you are working on a Linux system.

First, clone the repo:

```bash
git clone https://github.com/WilliamDormer/3DMenuVisualization.git
```

```bash
cd 3DMenuVisualization
```

Next, make sure to set the appropriate environment variables for CUDA:

```bash
CUDA_HOME=/usr/local/cuda
TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
```

Finally, create and activate the conda environment.

```bash
conda create --prefix ./env -f environment.yml
```

## Issues


