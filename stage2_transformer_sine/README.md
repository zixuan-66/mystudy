# Transformer Encoder for Sine Wave Forecasting


## 1. Project Introduction

This project implements a simplified Transformer Encoder for time series forecasting.

The task is:

Given previous 50 time steps of a sine wave,
predict the next 10 time steps.


The project aims to understand:

- Self-Attention
- Multi-Head Attention
- Positional Encoding
- Transformer Encoder
- Attention Visualization


---

## 2. Project Structure


stage2_transformer_sine
├── data
│   └── sine.py
├── models
│   ├── attention.py
│   ├── multi_head_attention.py
│   ├── positional_encoding.py
│   ├── encoder_block.py
│   └── transformer.py
├── checkpoints
│   └── best.pt
├── train.py
├── evaluate.py
├── visualize_attention.py
└── README.md


---

## 3. Dataset

Synthetic sine wave dataset:

\[
y = sin(x)
\]


Sliding window method is used.

Input:

50 historical points


Output:

10 future points


---

## 4. Model Architecture


The model contains:


### 1. Input Embedding

Maps input dimension:

1 → 64


### 2. Positional Encoding

Sinusoidal positional encoding is added
to provide temporal order information.


### 3. Transformer Encoder

Each Encoder Block contains:


Multi-Head Attention
↓
Residual Connection + LayerNorm
↓
Feed Forward Network
↓
Residual Connection + LayerNorm


### 4. Prediction Head


The final hidden state is mapped to:

10 future values


---

## 5. Training


Environment:

Python 3.10
PyTorch
wandb


Optimizer:

Adam


Loss:

MSELoss


Training:

Epochs: 50
Batch size: 64
Learning rate: 1e-3


---

## 6. Results


Training loss converges to:

~3e-5


Prediction example:


(Add your prediction figure here)


The model can accurately forecast
future sine wave values.


---

## 7. Attention Visualization


Attention weights are extracted from
Transformer Encoder.


Example:


(Add attention heatmap here)


The heatmap shows which historical
time steps the model focuses on when
encoding the sequence.


---

## 8. How to Run


Install dependencies:


```bash
conda activate ts_stage1
Train:
python train.py
Evaluate:
python evaluate.py
Visualize attention:
python visualize_attention.py