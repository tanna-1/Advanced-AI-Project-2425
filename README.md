# Advanced AI Project 24/25

## Abstract
This project explores different ways to scale MLPs, with the goal of developing an MLP-based network that can rival Transformer models.

The idea is inspired by the paper [Scaling MLPs: A Tale of Inductive Bias](https://arxiv.org/abs/2306.13575) where MLPs were shown to be capable of surpassing models with strong inductive bias given the optimal architecture.

[Optuna](https://optuna.org/) will be used to optimize the model architecture and the hyper-parameters.

The architecture of the model is based on [AUNNs](https://gwern.net/aunn). The model is designed to predict anything from text tokens to weights of another model.

This repository includes code to predict text tokens under the folder `text_prediction`.
It also includes code to predict weights of a larger CNN model using our model as a hypernet under the folder `hypernet`.

## Usage

### Setup

```bash
pip install .
```

### Training text prediction

```bash
advanced_ai_project optimize path/to/dataset.txt
advanced_ai_project train path/to/dataset.txt
```

### Evaluating text prediction

```bash
advanced_ai_project evaluate --start=0 --length=100
advanced_ai_project evaluate --start='-100'
advanced_ai_project autocomplete "Hi, my name is "
```

### Training hypernet

```bash
advanced_ai_project optimize_hypernet path/to/dataset.txt
advanced_ai_project train_hypernet path/to/cifar10 --num-epochs 200
```

### Evaluating hypernet

```bash
advanced_ai_project evaluate_hypernet path/to/cifar10
```

## References
- Alpha Dropout & SELU: [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- Inverted Bottleneck Block: [Scaling MLPs: A Tale of Inductive Bias](https://arxiv.org/abs/2306.13575)
- AUNNs: [Absolute Unit NNs: Regression-Based MLPs for Everything](https://gwern.net/aunn)
- HyperNetworks: [HyperNetworks](https://arxiv.org/abs/1609.09106)
