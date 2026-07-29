import torch
from torch.utils.data import Dataset, DataLoader


class SinDataset(Dataset):

    def __init__(self):

        # 创建数据
        self.x = torch.linspace(
            -3.14,
            3.14,
            2000
        )

        self.y = torch.sin(self.x)


    def __len__(self):

        # 数据数量
        return len(self.x)


    def __getitem__(self,index):

        # 根据索引返回一个样本
        return self.x[index], self.y[index]



dataset = SinDataset()


print("Dataset size:")
print(len(dataset))


print("First sample:")
print(dataset[0])

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


for batch_x,batch_y in dataloader:

    print(batch_x.shape)
    print(batch_y.shape)

    break