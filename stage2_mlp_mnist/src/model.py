import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        activation="relu",
        norm=None,
        dropout=0
    ):

        super().__init__()


        # ==========================
        # 1. 激活函数
        # ==========================

        if activation == "relu":

            act = nn.ReLU()


        elif activation == "sigmoid":

            act = nn.Sigmoid()


        elif activation == "tanh":

            act = nn.Tanh()


        elif activation == "gelu":

            act = nn.GELU()


        else:

            raise ValueError(
                "Unsupported activation"
            )



        # ==========================
        # 2. 构建网络
        # ==========================

        layers = []


        # -------- 第一层 --------

        layers.append(
            nn.Linear(784, 256)
        )


        # Norm

        if norm == "bn":

            layers.append(
                nn.BatchNorm1d(256)
            )


        elif norm == "ln":

            layers.append(
                nn.LayerNorm(256)
            )


        # Activation

        layers.append(
            act
        )


        # Dropout

        if dropout > 0:

            layers.append(
                nn.Dropout(dropout)
            )



        # -------- 第二层 --------

        layers.append(
            nn.Linear(256,128)
        )


        # Norm

        if norm == "bn":

            layers.append(
                nn.BatchNorm1d(128)
            )


        elif norm == "ln":

            layers.append(
                nn.LayerNorm(128)
            )


        # Activation

        layers.append(
            act
        )


        # Dropout

        if dropout > 0:

            layers.append(
                nn.Dropout(dropout)
            )



        # -------- 输出层 --------

        layers.append(
            nn.Linear(128,10)
        )



        # 组合网络

        self.net = nn.Sequential(
            *layers
        )



    # ==========================
    # forward
    # ==========================

    def forward(self,x):

        # MNIST:
        # [batch,1,28,28]
        #
        # 转为：
        # [batch,784]

        x = x.view(
            x.size(0),
            -1
        )


        return self.net(x)