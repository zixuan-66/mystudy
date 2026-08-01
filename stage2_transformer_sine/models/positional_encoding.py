import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=500
    ):

        super().__init__()


        pe = torch.zeros(
            max_len,
            d_model
        )


        # 位置
        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)


        # 分母
        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            )
            *
            (-math.log(10000.0) / d_model)
        )


        # 偶数位置
        pe[:,0::2] = torch.sin(
            position * div_term
        )


        # 奇数位置
        pe[:,1::2] = torch.cos(
            position * div_term
        )


        # 增加batch维度
        pe = pe.unsqueeze(0)


        self.register_buffer(
            "pe",
            pe
        )


    def forward(self,x):

        """
        x:
        batch, seq_len, d_model
        """

        x = x + self.pe[
            :,
            :x.size(1),
            :
        ]

        return x