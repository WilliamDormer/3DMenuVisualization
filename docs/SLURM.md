# SLURM


How to run the models on a slurm cluster.

Assumes the models have been installed correctly.

Request a GPU

```
srun --partition=gpunodes -c 4 --mem=12GB --gres=gpu:1 -t 60 --pty bash --login
```

Set environment variables

```bash
CUDA_HOME=/usr/local/cuda
TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
```


python train.py -s ../../data/nachos/ --eval -m output/horse_b
lender -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512