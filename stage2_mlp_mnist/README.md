# Stage2: MLP-MNIST Classification


## Overview

This project implements a Multi-Layer Perceptron (MLP) model for handwritten digit classification on the MNIST dataset using PyTorch.

The goal is to understand the complete deep learning training pipeline, including:

- Forward propagation
- Backward propagation
- Gradient descent optimization
- Model evaluation
- Experiment tracking with Weights & Biases


## Environment

- Python 3.10
- PyTorch
- torchvision
- wandb


## Dataset

Dataset:

MNIST

Task:

10-class handwritten digit classification


Input:

28 × 28 grayscale images


Output:

10 categories (0-9)


## Model Architecture

MLP architecture:

784 -> 256 -> 128 -> 10


The model contains:

- Fully connected layers
- Activation functions
- Optional normalization layers
- Optional dropout layers


## Training Configuration

Optimizer:

Adam


Loss:

Cross Entropy Loss


Batch size:

128


Learning rate:

0.001


Training epochs:

20


Experiment tracking:

Weights & Biases


## Experiments


## 1. Activation Function Comparison

Compared activation functions:

- ReLU
- Sigmoid
- Tanh
- GELU


Observation:

Different activation functions achieved similar performance on MNIST because the network is relatively shallow.


## 2. Normalization Comparison

Compared:

- No normalization
- Batch Normalization
- Layer Normalization


Purpose:

Study the effect of normalization techniques on training stability.


## 3. Dropout Experiment

Compared:

- Dropout 0
- Dropout 0.2
- Dropout 0.5


Purpose:

Analyze the effect of dropout on overfitting and generalization.


## 4. Weight Decay Experiment

Compared:

- weight_decay = 0
- weight_decay = 1e-4
- weight_decay = 1e-3


Purpose:

Study L2 regularization and its influence on model generalization.


## 5. Learning Rate Scheduler

Compared:

- None
- StepLR
- CosineAnnealingLR
- ReduceLROnPlateau


Purpose:

Analyze different learning rate adjustment strategies.


## Results

All experiments were recorded using Weights & Biases.

Project:

MLP-MNIST


Best configuration:

Activation:
ReLU
Normalization:
None
Dropout:
0
Weight Decay:
1e-4

(The final scheduler comparison is performed based on this configuration.)


## Project Structure

stage2_mlp_mnist
├── src
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── utils.py
│
├── checkpoints
│
├── data
│
└── README.md


## How to Run

Install dependencies:

```bash
pip install torch torchvision wandb
Run training:
python src/train.py
Experiment Tracking
All training experiments are logged with wandb.
The recorded metrics include:
training loss
validation loss
training accuracy
validation accuracy
learning rate