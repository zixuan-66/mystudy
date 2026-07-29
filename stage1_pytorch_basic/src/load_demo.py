import torch
from torch import nn


# =========================
# 重新定义模型结构
# =========================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(1,32),

            nn.ReLU(),

            nn.Linear(32,1)

        )


    def forward(self,x):

        return self.net(x)



# =========================
# 创建空模型
# =========================

model = MLP()



# =========================
# 加载参数
# =========================

model.load_state_dict(
    torch.load("sin_model.pth")
)



# 设置为推理模式

model.eval()



# =========================
# 测试
# =========================


x = torch.tensor(
    [[0.0]]
)


prediction = model(x)


print("Prediction:")
print(prediction.item())


print("True:")
print(torch.sin(x).item())