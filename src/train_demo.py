import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. 定义 Dataset
# =========================

class SinDataset(Dataset):

    def __init__(self):
        """
        初始化数据
        """

        # 输入数据 x
        self.x = torch.linspace(
            -3.14,
            3.14,
            2000
        ).reshape(-1, 1)


        # 标签 y = sin(x)
        self.y = torch.sin(self.x)


    def __len__(self):
        """
        返回数据数量
        """

        return len(self.x)



    def __getitem__(self, index):
        """
        根据索引返回一个样本
        """

        return self.x[index], self.y[index]



# =========================
# 2. 定义模型
# =========================

class MLP(nn.Module):

    def __init__(self):
        """
        定义网络结构
        """

        super().__init__()


        self.net = nn.Sequential(

            # 输入1维，输出32维
            nn.Linear(1, 32),

            # 激活函数
            nn.ReLU(),

            # 32维输入，输出1维
            nn.Linear(32, 1)
        )



    def forward(self, x):
        """
        定义前向传播过程
        """

        return self.net(x)



# =========================
# 3. 设置设备
# =========================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("Using device:", device)



# =========================
# 4. 创建 Dataset 和 DataLoader
# =========================


dataset = SinDataset()


dataloader = DataLoader(

    dataset,

    batch_size=32,

    shuffle=True
)



# =========================
# 5. 创建模型
# =========================


model = MLP()


# 放到 M5 GPU
model = model.to(device)



# =========================
# 6. 定义损失函数和优化器
# =========================


criterion = nn.MSELoss()


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001

)



# =========================
# 7. 完整训练循环
# =========================


epochs = 100


for epoch in range(epochs):


    total_loss = 0



    # DataLoader提供batch数据

    for batch_x, batch_y in dataloader:


        # 数据移动到设备

        batch_x = batch_x.to(device)

        batch_y = batch_y.to(device)



        # -------------------------
        # 1. Forward
        # -------------------------

        y_pred = model(batch_x)



        # -------------------------
        # 2. Loss
        # -------------------------

        loss = criterion(
            y_pred,
            batch_y
        )



        # -------------------------
        # 3. 清空梯度
        # -------------------------

        optimizer.zero_grad()



        # -------------------------
        # 4. Backward
        # -------------------------

        loss.backward()



        # -------------------------
        # 5. 更新参数
        # -------------------------

        optimizer.step()



        total_loss += loss.item()



    # 每10轮打印一次

    if epoch % 10 == 0:

        print(
            f"Epoch [{epoch}/{epochs}], Loss: {total_loss:.6f}"
        )



# =========================
# 8. 测试模型
# =========================


test_x = torch.tensor(
    [[0.0]],
    dtype=torch.float32
).to(device)



prediction = model(test_x)



print("\nTest:")
print("x = 0")


print(
    "Prediction:",
    prediction.item()
)


print(
    "True value:",
    torch.sin(torch.tensor(0.)).item()
)

# =========================
# 保存模型
# =========================

torch.save(
    model.state_dict(),
    "sin_model.pth"
)


print("Model saved!")