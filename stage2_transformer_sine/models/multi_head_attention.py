import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads
    ):

        super().__init__()


        assert d_model % num_heads == 0


        self.d_model = d_model

        self.num_heads = num_heads


        # 每个head的维度
        self.head_dim = d_model // num_heads


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


        self.out_linear = nn.Linear(
            d_model,
            d_model
        )


    def forward(self,x):

        batch_size = x.size(0)


        # Q K V
        Q = self.q_linear(x)

        K = self.k_linear(x)

        V = self.v_linear(x)



        # 分成多个head
        Q = Q.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        )


        K = K.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        )


        V = V.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        )


        # 调整维度
        # batch, head, seq, dim

        Q = Q.transpose(1,2)

        K = K.transpose(1,2)

        V = V.transpose(1,2)



        # attention score

        scores = torch.matmul(
            Q,
            K.transpose(-2,-1)
        )


        scores = scores / math.sqrt(
            self.head_dim
        )


        attention = torch.softmax(
            scores,
            dim=-1
        )


        output = torch.matmul(
            attention,
            V
        )


        # 合并head

        output = output.transpose(
            1,2
        )


        output = output.contiguous().view(
            batch_size,
            -1,
            self.d_model
        )


        output = self.out_linear(
            output
        )


        return output, attention