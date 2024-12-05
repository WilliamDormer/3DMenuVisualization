## Troubleshooting

Note that the gaussian splatting models in this repo use updated dependencies compared to the original implementations.
Additionally, some changes were made to the source code to fix bugs that caused installation issues.

### torch.OutOfMemoryError

This can occur if you forgot to set the `TORCH_CUDA_ARCH_LIST` environment variable.
The submodules of the gaussian splatting models have inflated memory usage due to CUDA compatibility issues.

To resolve this, first `cd` into one of the gaussian splatting model folders, say 'gaussian-splatting', and run the following commands (while your virtual environment is active):

```bash
cd submodules

python3 -m pip uninstall diff_gaussian_rasterization_3dgs
python3 diff-gaussian-rasterization/setup.py clean

python3 -m pip uninstall simple_knn_3dgs
python3 simple-knn/setup.py clean

pyhon3 -m pip uninstall fused_ssim
python3 fused-ssim/setup.py clean


TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX" pip install diff-gaussian-rasterization/
TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX" pip install simple-knn/
TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX" pip install fused-ssim/
```

Repeat for each of the models you wish to resolve.
Refer to [https://github.com/graphdeco-inria/gaussian-splatting/issues/99](https://github.com/graphdeco-inria/gaussian-splatting/issues/99), for a more complete discussion of the issue.

### cudaErrorIllegalMemoryAccess

This can occur when there is not enough memory available in your GPU.
A few ways to resolve this include:

1. Reducing the size of your inputs (i.e. use smaller images for training)
2. Use fewer training iterations

There are other possible causes too. The issue may be with COLMAP (Refer to [https://colmap.github.io/faq.html#feature-matching-fails-due-to-illegal-memory-access](https://colmap.github.io/faq.html#feature-matching-fails-due-to-illegal-memory-access) for more details).

### IndexError: list index out or range

This can also be caused by incorrectly setting the `TORCH_CUDA_ARCH_LIST` variable.
You may encounter an error like:

```bash
...
    arch_list[-1] += '+PTX'
      IndexError: list index out of range
```

In which case you need to reinstall the submodules with the appropriate settings for `TORCH_CUDA_ARCH_LIST`. See above for instructions.

### RuntimeError(CUDA_MISMATCH_MESSAGE)

```bash
raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
      RuntimeError:
      The detected CUDA version (12.6) mismatches the version that was used to compile
      PyTorch (11.8). Please make sure to use the same CUDA versions.
```

Depending on when you are using this repo, you may run into CUDA compatibility issues.
Typically, Pytorch comes with a CUDA SDK so the code shouldn't be relying on your system's CUDA files.
Yet the submodules expect the `CUDA_HOME` variable to be set, so make sure it's pointing to your CUDA files (like `/usr/local/cuda`)

### ImportError: gaussian_splatting/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent

We ran into issues when using `cudatoolkit` (version 11), `pytorch` (version 1) and earlier versions of `python` (3.8) as indicated in the original model repositories.

The error went away when we upgraded to `python3.10` and `pytorch` (version 2) and `pytorch-cuda` (version 12.4)