import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        self.q_linear = nn.Linear(
            d_model,
            d_model
        )

        self.k_linear = nn.Linear(
            d_model,
            d_model
        )

        self.v_linear = nn.Linear(
            d_model,
            d_model
        )


    def forward(self, x):

        # x:
        # batch, seq_len, d_model


        Q = self.q_linear(x)

        K = self.k_linear(x)

        V = self.v_linear(x)


        # QK^T
        scores = torch.matmul(
            Q,
            K.transpose(-2,-1)
        )


        # scaled
        scores = scores / math.sqrt(
            Q.size(-1)
        )


        # attention weight
        attention = torch.softmax(
            scores,
            dim=-1
        )


        # weighted sum
        output = torch.matmul(
            attention,
            V
        )


        return output, attention