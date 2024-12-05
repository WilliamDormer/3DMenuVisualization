# Benchmarks

Performance of Gaussian Splatting models averaged over 5 scenes from the MetaFood3D dataset (Nachos, Grapes, Hotdog, Pasta, Pizza).


| Model                     | LPIPS $\downarrow$ | SSIM $\uparrow$ | PSNR $\uparrow$ | Size (MB) $\downarrow$ | \# Splats | Train Iters | Train Time (min) $\downarrow$ |
|---------------------------|--------------------|-----------------|-----------------|------------------------|-----------|-------------|-------------------------------|
| Gaussian Splatting        | 0.219              | 0.920           | 30.72           | 89                     | 374126    | 30000       | **17.4**                 |
| Self-Organizing Gaussians | **0.199**     | **0.926**  | **31.075** | **30.61**         | 363931    | 30000       | 26.2                          |
| FSGS                      | 0.298              | 0.875           | 26.75           | 384.8                  | 1551635   | 10000       | 21.4                          |
| Gaussian Shader           | 0.390              | 0.854           | 22.31           | 45.58*                 | 514411    | 10000*      | 120                           |