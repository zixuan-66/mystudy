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


x = x.unsqueeze(0).to(device)



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

    pred, attention = model(x)



# 取第1层 Encoder
# 第1个head

attn = attention[0]


attn = attn[0,0]


attn = attn.cpu()



print(attn.shape)



# 画图

plt.figure(figsize=(8,6))


plt.imshow(
    attn,
    cmap="viridis"
)


plt.colorbar()


plt.xlabel(
    "Key timestep"
)

plt.ylabel(
    "Query timestep"
)


plt.title(
    "Attention Weight Heatmap"
)


plt.show()