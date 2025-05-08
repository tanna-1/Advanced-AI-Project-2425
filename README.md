# Advanced AI Project 24/25

## Abstract
This project explores different ways to scale MLPs, with the goal of developing an MLP-based network that can rival Transformer models.

The idea is inspired by the paper [Scaling MLPs: A Tale of Inductive Bias](https://arxiv.org/abs/2306.13575) where MLPs were shown to be capable of surpassing models with strong inductive bias given the optimal architecture.

[Optuna](https://optuna.org/) will be used to optimize the model architecture and the hyper-parameters.

The architecture of the model is based on [AUNNs](https://gwern.net/aunn). The model is designed to predict anything from text tokens to weights of another model.

This repository includes code to predict text tokens under the folder `text_prediction`.
It also includes code to predict weights of a larger CNN model using our model as a [hypernetwork](https://arxiv.org/abs/1609.09106) under the folder `hypernet`.

## Usage

### Setup

```bash
pip install "torch~=2.7.0" "torchvision~=0.22.0" torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install .
```

### Notebooks

Follow these steps to work with the project notebooks:

1. **Prepare the text prediction dataset**
   - Run the `text_prediction_data` notebook before running the `text_prediction` notebook.
   - This is not necessary for the `hypernet` notebook as it automatically downloads the dataset.

2. **Open a notebook**
   - Navigate to the `notebooks` folder and open any notebook you wish to run.

4. **Resume existing work**
   - Notebooks automatically resume training and optimization if study/checkpoint files exist.

5. **Modify configuration or start fresh**
   - To change settings: Edit the configuration section within each notebook.
   - To start from scratch: Delete any existing checkpoint or study files.

## References
- Alpha Dropout & SELU: [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- Inverted Bottleneck Block: [Scaling MLPs: A Tale of Inductive Bias](https://arxiv.org/abs/2306.13575)
- AUNNs: [Absolute Unit NNs: Regression-Based MLPs for Everything](https://gwern.net/aunn)
- HyperNetworks: [HyperNetworks](https://arxiv.org/abs/1609.09106)
