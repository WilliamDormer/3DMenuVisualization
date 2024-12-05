# Benchmarks

## MetaFood3D

Performance of Gaussian Splatting models averaged over 5 scenes from the MetaFood3D dataset (Nachos, Grapes, Hotdog, Pasta, Pizza).

| Model                     | LPIPS $\downarrow$ | SSIM $\uparrow$ | PSNR $\uparrow$ | Size (MB) $\downarrow$ | \# Splats | Train Iters | Train Time (min) $\downarrow$ |
|---------------------------|--------------------|-----------------|-----------------|------------------------|-----------|-------------|-------------------------------|
| Gaussian Splatting        | 0.219              | 0.920           | 30.72           | 89                     | 374126    | 30000       | **17.4**                 |
| Self-Organizing Gaussians | **0.199**     | **0.926**  | **31.075** | **30.61**         | 363931    | 30000       | 26.2                          |
| FSGS                      | 0.298              | 0.875           | 26.75           | 384.8                  | 1551635   | 10000       | 21.4                          |
| Gaussian Shader           | 0.390              | 0.854           | 22.31           | 45.58*                 | 514411    | 10000*      | 120                           |

## DormerFood

For our custom dataset, DormerFood, we benchmarked the best performing model, Self-Organizing Gaussians.
The dataset can be downloaded [here](/):

| Scene              | LPIPS ↓ | SSIM ↑ | PSNR ↑ | Output .ply File Size (MB) | # of Gaussian Splats | # input images | # Colmap reconstructed images | Training Iterations | Train Time |
|--------------------|---------|--------|--------|----------------------------|----------------------|----------------|-------------------------------|---------------------|------------|
| Baseline           |   0.074 |  0.985 |  38.65 |                        6.4 |                72361 |            249 |                           249 |               30000 | 17 min     |
| Visible background |   0.123 |  0.965 |  34.30 |                       15.7 |               187489 |            213 |                           213 |               30000 | 18 min     |
| Close up           |   0.089 |  0.967 |  36.40 |                       13.6 |               147456 |            243 |                           243 |               30000 | 17 min     |
| cluttered          |   0.416 |  0.670 |  18.60 |                        6.0 |                90601 |            260 |                             2 |               30000 | 10 min     |
| erratic            |   0.295 |  0.840 |  23.01 |                        9.1 |               109561 |            230 |                            13 |               30000 | 8 min      |
| occluded           |   0.099 |  0.971 |  35.94 |                       14.0 |               145924 |            256 |                           256 |               30000 | 19 min     |
| spotlight          |   0.092 |  0.978 |  35.94 |                        8.5 |                97344 |            239 |                           239 |               30000 | 16 min     |
| zoom in            |   0.119 |  0.964 |  32.76 |                       10.3 |               104976 |            213 |                           213 |               30000 | 18 min     |