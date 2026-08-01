import torch
import matplotlib.pyplot as plt

from data.sine import SineDataset
from models.transformer import TransformerForecast



device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)



# 数据

dataset = SineDataset()


x,y = dataset[100]


# 增加 batch 维度

x_input = x.unsqueeze(0).to(device)



# 模型

model = TransformerForecast().to(device)


model.load_state_dict(
    torch.load(
        "checkpoints/best.pt",
        map_location=device
    )
)


model.eval()



with torch.no_grad():

    pred, attention = model(
        x_input
    )



pred = pred.squeeze().cpu()


y = y.cpu()



# 画图

plt.figure(figsize=(10,4))


plt.plot(
    range(50),
    x.squeeze(),
    label="history"
)


plt.plot(
    range(50,60),
    y,
    label="real future"
)


plt.plot(
    range(50,60),
    pred,
    label="prediction"
)


plt.legend()

plt.title(
    "Transformer Sine Forecast"
)


plt.show()