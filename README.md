# Advanced AI Project 24/25

## Abstract
This project explores different ways to scale MLPs, with the goal of developing an MLP-based network that can rival Transformer models.

The idea is inspired by the paper [Scaling MLPs: A Tale of Inductive Bias](https://arxiv.org/abs/2306.13575) where MLPs were shown to be capable of surpassing models with strong inductive bias given the optimal architecture.

[Optuna](https://optuna.org/) will be used to optimize the model architecture and the hyper-parameters.

The model will also have the ability to input/output latent tensors, so as to be multimodal by repurposing the latent output as image output via a CNN-based autoencoder.

## Usage

### Setup

```bash
cd llm
pip install .
```

### Training

```bash
advanced_ai_project optimize path/to/dataset.txt --num-epochs 2
advanced_ai_project train path/to/dataset.txt --num-epochs 2
```

### Running

```bash
advanced_ai_project evaluate --start='-100'
advanced_ai_project autocomplete "Hi, my name is "
```

## References
- Alpha Dropout & SELU: [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- Inverted Bottleneck Block: [Scaling MLPs: A Tale of Inductive Bias](https://arxiv.org/abs/2306.13575)
