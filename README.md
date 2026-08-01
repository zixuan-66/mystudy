# Deep Learning Study Repository

这是一个用于记录深度学习学习过程、模型复现和实验实践的 PyTorch 项目仓库。

主要目标是通过代码实现和实验分析，逐步掌握深度学习基础知识、模型结构以及训练流程。


---

# Repository Structure


study
├── stage1_pytorch_basic
├── stage2_mlp_mnist
└── stage2_transformer_sine


---

# Completed Projects


## Stage 1: PyTorch Basic

路径：

stage1_pytorch_basic


主要学习 PyTorch 基础：

- Tensor 基本操作
- Autograd 自动求导
- nn.Module 网络构建
- Dataset & DataLoader
- 完整训练流程
- 模型保存与加载


通过简单实验理解深度学习训练流程：

Forward
 ↓
Loss
 ↓
Backward
 ↓
Optimizer Step


---

## Stage 2: Deep Learning Practice


### 1. MLP for MNIST Classification

路径：

stage2_mlp_mnist


使用多层感知机完成 MNIST 手写数字分类任务。


主要内容：

- MLP 网络搭建
- 激活函数对比
- BatchNorm / LayerNorm
- Dropout
- Weight Decay
- Learning Rate Scheduler
- wandb 实验记录



---

### 2. Transformer Encoder for Sine Forecasting

路径：

stage2_transformer_sine


实现简化版 Transformer Encoder，并完成正弦波时间序列预测任务。


主要内容：

- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Transformer Encoder Block
- 时间序列预测
- Attention 权重可视化


---

# Experiment Tracking

训练实验使用：

- Weights & Biases (wandb)


记录：

- Training Loss
- Validation Loss
- Accuracy
- Learning Rate


---

# Environment


主要环境：

Python 3.10
PyTorch
wandb


运行平台：

- macOS
- Apple Silicon MPS


---

# Future Plan


后续学习方向：

- Time Series Forecasting
- CNN / ResNet
- Vision Transformer
- Large Language Model
- LoRA Fine-tuning
- Advanced Transformer Models


---

# Author

Zixuan Zhou