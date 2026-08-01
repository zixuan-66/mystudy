import torch
import torch.nn as nn

from models.multi_head_attention import MultiHeadAttention



class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        hidden_dim
    ):

        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                d_model,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                d_model
            )
        )


    def forward(self,x):

        return self.net(x)





class EncoderBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        hidden_dim,
        dropout=0.1
    ):

        super().__init__()


        # Multi-head Attention

        self.attention = MultiHeadAttention(
            d_model,
            num_heads
        )


        # FFN

        self.ffn = FeedForward(
            d_model,
            hidden_dim
        )


        # LayerNorm

        self.norm1 = nn.LayerNorm(
            d_model
        )


        self.norm2 = nn.LayerNorm(
            d_model
        )


        self.dropout = nn.Dropout(
            dropout
        )



    def forward(self,x):


        # ===== Attention =====

        attention_output, attention_weight = self.attention(x)


        # Residual + Norm

        x = self.norm1(
            x + self.dropout(attention_output)
        )



        # ===== FFN =====

        ffn_output = self.ffn(x)



        # Residual + Norm

        x = self.norm2(
            x + self.dropout(ffn_output)
        )


        return x, attention_weight