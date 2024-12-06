# Parameters

Below we list the parameters that are shared between the models, but differ in their default values:

| Parameter Name | Parameter Type     | Symbol in Code         | Description                                                                                                                                                                                                                      | GShader | FSGS     | 3DGS   | SOGS   |
|----------------|--------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|----------|--------|--------|
| Images                | ModelParams        | _images                |                                                                                                                                                                                                                                  | images  | images_8 | images | images |
| Densification Gradient Threshold | OptimizationParams | densify_grad_threshold | The threshold at which gaussians are 'densified' (i.e. split and cloned) during gradient descent. A lower threshold means the gaussians are split when the gradient of their xyz position has accumulated a lower gradient value |  0.0002 |   0.0005 | 0.0002 | 0.0002 |
|Densify Until Iteration| OptimizationParams | densify_until_iter     | Determines how often the densification (splitting of Gaussians) occurs during training, based on iterations                                                                                                                      | 15_000  | 10_000   | 15_000 | 15_000 |
|Iterations| OptimizationParams | iterations             | The number of iterations to train the model                                                                                                                                                                                      | 30_000  | 10_000   | 30_000 | 30_000 |
|Opacity Learning Rate| OptimizationParams | opacity_lr             | Opacity learning rate for placing gaussian splats                                                                                                                                                                                                            |    0.05 |     0.05 |  0.025 |   0.05 |
|Position Learning| OptimizationParams | position_lr_max_steps  | Position learning rate for placing gaussian splats                                                                                                                                                                                                           | 30_000  | 10_000   | 30_000 | 30_000 |
|Scaling Learning Rate| OptimizationParams | scaling_lr             | Scaling factor for learning rate when optimizing placement of splats                                                                                                                                                             |   0.001 |    0.005 |  0.005 |  0.005 |


For further discussion of the remaining shared parameters, or model-specific parameters, refer to each paper:

