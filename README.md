# PyTorch Stage1 Learning

This repository records my learning process of PyTorch fundamentals, including tensor operations, automatic differentiation, model construction, data loading, training workflow, and model persistence.

## Environment

- macOS
- Python 3.10
- PyTorch
- Miniconda
- Apple Silicon MPS acceleration


## Project Structure

.
├── README.md
├── environment.yml
├── src
│   ├── tensor_demo.py
│   ├── autograd_demo.py
│   ├── module_demo.py
│   ├── dataloader_demo.py
│   ├── train_demo.py
│   └── save_load_demo.py
│
├── docs
│   └── learning_notes.md
│
└── checkpoints

## Learning Contents


### 1. Tensor Operations

Learned basic PyTorch tensor operations:

- Tensor creation
- Tensor indexing and slicing
- Broadcasting
- Device management
- MPS acceleration


### 2. Automatic Differentiation

Explored PyTorch autograd mechanism:

- requires_grad
- computational graph
- backward propagation
- gradient calculation


### 3. Neural Network Construction

Learned how to build models with:

- torch.nn.Module
- __init__()
- forward()
- trainable parameters


### 4. Dataset and DataLoader

Implemented data loading pipeline:

Dataset
    ↓
DataLoader
    ↓
Batch training

Learned:

- Custom Dataset
- __len__()
- __getitem__()
- mini-batch loading


### 5. Complete Training Pipeline

Implemented the standard deep learning workflow:

Input Data
↓
Forward
↓
Loss Calculation
↓
Backward
↓
Optimizer Step
↓
Parameter Update


### 6. Model Saving and Loading

Learned model persistence:

- torch.save()
- torch.load()
- state_dict()

Models can be saved after training and loaded for inference.


## Training Framework

The overall PyTorch training process:

Prepare Dataset
↓
Create DataLoader
↓
Define Model
↓
Forward Pass
↓
Calculate Loss
↓
Backward Propagation
↓
Update Parameters
↓
Save Model


## Purpose

This project provides a foundation for further study of:

- Deep Learning
- Time Series Modeling
- Transformer Models
- AI Research Projects