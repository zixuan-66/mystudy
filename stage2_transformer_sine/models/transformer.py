import torch
import torch.nn as nn


from models.positional_encoding import PositionalEncoding
from models.encoder_block import EncoderBlock



class TransformerForecast(nn.Module):

    def __init__(
        self,
        input_dim=1,
        d_model=64,
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
        pred_len=10
    ):

        super().__init__()


        self.pred_len = pred_len


        # 输入映射
        # 1维 -> 64维

        self.embedding = nn.Linear(
            input_dim,
            d_model
        )


        # 位置编码

        self.position_encoding = PositionalEncoding(
            d_model
        )


        # 多层Encoder

        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    num_heads,
                    hidden_dim
                )
                for _ in range(num_layers)
            ]
        )


        # 输出层

        self.output_layer = nn.Linear(
            d_model,
            pred_len
        )



    def forward(self,x):


        # x:
        # batch,50,1


        x = self.embedding(x)


        # 加位置

        x = self.position_encoding(x)



        attention_weights = []


        # Encoder

        for layer in self.encoder_layers:

            x, attention = layer(x)

            attention_weights.append(
                attention
            )



        # 取最后一个时间点

        x = x[:,-1,:]


        # 预测未来10步

        output = self.output_layer(x)



        return output, attention_weights