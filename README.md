# Deep Learning Study Repository

这是一个用于记录个人深度学习学习过程、实验复现以及模型实践的 PyTorch 项目仓库。

主要目标是通过实际代码实现，系统学习深度学习基础理论、模型结构、训练技巧以及常见工程实践。

所有实验均基于 PyTorch 实现，并使用 Weights & Biases（wandb）进行实验记录和结果分析。


---

# Repository Structure

study
├── stage1_pytorch_basic
│
├── stage2_mlp_mnist
│
└── README.md


---

# Learning Roadmap


## Stage 1: PyTorch Basic

路径：

stage1_pytorch_basic


主要学习 PyTorch 基础组件，包括：


### Tensor

- 张量基本操作
- Tensor shape变化
- GPU/MPS设备迁移


### Autograd

- 自动求导机制
- computational graph
- backward传播


### Neural Network Module

- nn.Module使用
- Linear层
- 参数管理


### Dataset & DataLoader

- 数据集构建
- batch训练
- 数据加载流程


### Training Pipeline

完整训练流程：

Data
 ↓
Forward
 ↓
Loss
 ↓
Backward
 ↓
Optimizer Step


---

# Stage 2: MLP-MNIST Classification

路径：

stage2_mlp_mnist


基于多层感知机（MLP）完成 MNIST 手写数字分类任务。


## Model

网络结构：

784 → 256 → 128 → 10


实现内容：

- 前向传播
- 反向传播
- 梯度下降优化
- 模型保存与加载


---

## Experiments


### 1. Activation Function

比较不同激活函数：

- ReLU
- Sigmoid
- Tanh
- GELU


分析：

- 梯度传播
- 收敛速度
- 分类性能


---

### 2. Normalization

比较：

- Batch Normalization
- Layer Normalization
- No Normalization


研究归一化方法对训练稳定性的影响。


---

### 3. Regularization


### Dropout

实验不同 dropout 比例：

- 0
- 0.2
- 0.5


分析：

- 过拟合问题
- 泛化能力


### Weight Decay

实验：

- 0
- 1e-4
- 1e-3


研究 L2 正则化对模型性能的影响。


---

### 4. Learning Rate Scheduler

比较：

- None
- StepLR
- CosineAnnealingLR
- ReduceLROnPlateau


分析不同学习率调整策略对训练过程的影响。


---

# Experiment Tracking

所有训练实验均使用：

## Weights & Biases (wandb)


记录指标：

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Learning Rate


实验结果可在 wandb 项目中查看。


---

# Development Environment


主要环境：

Python 3.10
PyTorch
torchvision
wandb


运行环境：

- macOS
- Apple Silicon MPS acceleration


---

# Engineering Practice


项目中包含：

- Git版本管理
- Conda环境管理
- 实验日志管理
- 模型Checkpoint保存
- README文档维护


---

# Future Plan


后续学习方向：


## Computer Vision

- CNN
- ResNet
- Vision Transformer


## Sequence Model

- RNN
- LSTM
- Transformer


## Large Language Model

- Transformer结构
- Attention机制
- LoRA微调
- LLM训练与推理


## Advanced Topics

- Distributed Training
- Model Optimization
- AI System Engineering


---

# Author

Zixuan Zhou


This repository records my learning journey in deep learning and artificial intelligence.