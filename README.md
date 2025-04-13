# Advanced AI Project 24/25

## Abstract
This project explores different ways to scale MLPs, with the goal of developing an MLP-based network that can rival Transformer models.

The idea is inspired by the paper [Scaling MLPs: A Tale of Inductive Bias](https://arxiv.org/abs/2306.13575) where MLPs were shown to be capable of surpassing models with strong inductive bias given the optimal architecture.

[Optuna](https://optuna.org/) will be used to optimize the model architecture and the hyper-parameters.

## Usage

1. Optimize hyperparameters: `$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"; advanced_ai_project optimize [dataset_path]`
2. Train model: `advanced_ai_project train [dataset_path]`
3. Evaluate model: `advanced_ai_project evaluate`

## References
- Alpha Dropout & SELU: [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- Inverted Bottleneck Block: [Scaling MLPs: A Tale of Inductive Bias](https://arxiv.org/abs/2306.13575)
