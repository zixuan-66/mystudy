import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb


from data.sine import SineDataset
from models.transformer import TransformerForecast



# =====================
# 参数
# =====================

epochs = 50

batch_size = 64

lr = 1e-3



device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)


print("device:", device)



# =====================
# wandb
# =====================

wandb.init(
    project="transformer-sine",
    name="baseline-transformer",
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "model": "TransformerEncoder"
    }
)



# =====================
# Dataset
# =====================

dataset = SineDataset()


loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)



# =====================
# Model
# =====================

model = TransformerForecast().to(device)



# =====================
# Loss + Optimizer
# =====================

criterion = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr
)



# =====================
# Training
# =====================

best_loss = float("inf")


for epoch in range(epochs):


    model.train()


    total_loss = 0



    for x,y in loader:


        x = x.to(device)

        y = y.to(device)



        # forward

        pred, attention = model(x)



        loss = criterion(
            pred,
            y
        )



        # backward

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        total_loss += loss.item()



    avg_loss = total_loss / len(loader)



    print(
        f"Epoch {epoch+1}/{epochs}, Loss:{avg_loss:.6f}"
    )


    wandb.log(
        {
            "train_loss":avg_loss,
            "epoch":epoch
        }
    )



    # 保存最佳模型

    if avg_loss < best_loss:

        best_loss = avg_loss


        torch.save(
            model.state_dict(),
            "checkpoints/best.pt"
        )



wandb.finish()