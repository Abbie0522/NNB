# Bank Term Deposit Subscription Prediction Using MLP

## Project Overview

This project uses a Multilayer Perceptron (MLP) neural network to predict whether bank customers will subscribe to a term deposit. The model is implemented using PyTorch and trained on the UCI Bank Marketing Dataset.

## Dataset

- Dataset: UCI Bank Marketing Dataset
- Samples: 45,211
- Target: Term deposit subscription (yes/no)
- Input features: 15 original features, expanded to 50 after preprocessing
- Data split: 70% training, 15% validation, 15% testing

## Neural Network Architecture

- Input layer: 50 features
- Hidden layer 1: 64 neurons with ReLU
- Hidden layer 2: 32 neurons with ReLU
- Output layer: 1 neuron
- Loss function: BCEWithLogitsLoss
- Optimizer: Adam

## Experiments

Three experiments were conducted:

1. Baseline: Learning rate 0.001, batch size 32
2. Experiment 1: Learning rate 0.0005, batch size 32
3. Experiment 2: Learning rate 0.001, batch size 64

Each experiment runs for 30 epochs. The best model is selected using validation F1-score.

## Docker Training

Build the training image:

```powershell
docker build -f Docker/Dockerfile.train -t bank-mlp-train .
```

Run the training container:

```powershell
docker run --rm -v "${PWD}/Outputs:/app/Outputs" -v "${PWD}/Model:/app/Model" bank-mlp-train
```

## Docker Testing

Build the testing image:

```powershell
docker build -f Docker/Dockerfile.test -t bank-mlp-test .
```

Run the testing container:

```powershell
docker run --rm -v "${PWD}/Outputs:/app/Outputs" -v "${PWD}/Model:/app/Model:ro" bank-mlp-test
```

## Outputs

The training process saves the trained model, preprocessing pipeline, processed datasets, and experiment histories. The testing process generates evaluation metrics and a confusion matrix.

## Reproducibility

The project uses a fixed random seed of 42 and pinned Python dependencies. The training and testing containers can be built and executed independently. Run the training container before the testing container to generate the required model and processed data.
