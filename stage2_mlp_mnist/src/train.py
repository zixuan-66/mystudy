import torch
import torch.nn as nn
import wandb
import os

from dataset import get_data
from model import MLP



# =====================
# 验证函数
# =====================

def evaluate(model, loader, loss_fn, device):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)


            output = model(x)


            loss = loss_fn(
                output,
                y
            )


            total_loss += loss.item()


            pred = output.argmax(dim=1)


            correct += (
                pred == y
            ).sum().item()


            total += y.size(0)



    avg_loss = total_loss / len(loader)

    acc = correct / total


    return avg_loss, acc



# =====================
# Early Stopping
# =====================

class EarlyStopping:


    def __init__(self, patience=5):

        self.patience = patience

        self.counter = 0

        self.best_loss = None


    def step(self, val_loss):

        if self.best_loss is None:

            self.best_loss = val_loss

            return False


        if val_loss > self.best_loss:

            self.counter += 1


            if self.counter >= self.patience:

                return True


        else:

            self.best_loss = val_loss

            self.counter = 0


        return False





# =====================
# 设备
# =====================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)


os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


print("device:", device)



# =====================
# wandb
# =====================


wandb.init(

    project="MLP-MNIST",

    name="scheduler_plateau",

    config={

        "model":"MLP",

        "dataset":"MNIST",

        "epochs":20,

        "batch_size":128,

        "lr":0.001,

        "optimizer":"Adam",

        "activation":"relu",

        "norm":"none",

        "dropout":0,

        "weight_decay":1e-4

    }

)



config = wandb.config



# =====================
# 数据
# =====================


train_loader,val_loader,test_loader = get_data(
    config.batch_size
)



# =====================
# 模型
# =====================

model = MLP(
    activation="relu",
    norm="none",
    dropout=0
).to(device)



# =====================
# loss
# =====================


loss_fn = nn.CrossEntropyLoss()



# =====================
# optimizer
# =====================


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=config.lr,

    weight_decay=1e-4,

)




# =====================
# scheduler
# =====================


# scheduler=torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=5,
#     gamma=0.5
# )

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=20
# )

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# =====================
# early stop
# =====================


early_stop = EarlyStopping(
    patience=5
)



# 保存最佳模型

best_val_acc = 0



# =====================
# train
# =====================


for epoch in range(config.epochs):


    model.train()


    total_loss = 0

    correct = 0

    total = 0



    for x,y in train_loader:


        x=x.to(device)

        y=y.to(device)



        # forward

        output=model(x)



        loss=loss_fn(
            output,
            y
        )



        # 清空梯度

        optimizer.zero_grad()



        # backward

        loss.backward()



        # 更新参数

        optimizer.step()



        total_loss += loss.item()



        pred = output.argmax(dim=1)


        correct += (
            pred==y
        ).sum().item()


        total += y.size(0)



    # train指标


    train_loss = (
        total_loss /
        len(train_loader)
    )


    train_acc = (
        correct /
        total
    )



    # validation


    val_loss,val_acc = evaluate(

        model,

        val_loader,

        loss_fn,

        device

    )

    # =====================
    # scheduler
    # =====================
    
    if scheduler is not None:

      if isinstance(
        scheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau
       ):

        scheduler.step(val_loss)

      else:

        scheduler.step()

    



    # wandb记录

    wandb.log({

        "epoch":epoch,

        "train_loss":train_loss,

        "train_acc":train_acc,

        "val_loss":val_loss,

        "val_acc":val_acc

    })



    print(
        f"Epoch [{epoch+1}/{config.epochs}]",
        f"Train Loss:{train_loss:.4f}",
        f"Train Acc:{train_acc:.4f}",
        f"Val Loss:{val_loss:.4f}",
        f"Val Acc:{val_acc:.4f}"
    )



    # 保存最佳模型

    if val_acc > best_val_acc:


        best_val_acc = val_acc


    torch.save(
    model.state_dict(),
    os.path.join(
        CHECKPOINT_DIR,
        "best_model.pt"
    )
)


    print(
            "save best model:",
            best_val_acc
        )



    # early stopping

    if early_stop.step(val_loss):

        print("Early stopping!")

        break



wandb.finish()