import torch
from torch.utils.data import Dataset


class SineDataset(Dataset):

    def __init__(
        self,
        seq_len=50,
        pred_len=10
    ):

        self.seq_len = seq_len
        self.pred_len = pred_len


        x = torch.linspace(
            0,
            100,
            10000
        )

        y = torch.sin(x)


        self.samples = []


        for i in range(
            len(y)-seq_len-pred_len
        ):

            input_seq = y[
                i:i+seq_len
            ]

            target = y[
                i+seq_len:
                i+seq_len+pred_len
            ]


            self.samples.append(
                (
                    input_seq.unsqueeze(-1),
                    target
                )
            )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self,index):

        return self.samples[index]