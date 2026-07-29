import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split



def get_data(batch_size):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])


    # 当前项目根目录
    # stage2_mlp_mnist/
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )


    DATA_DIR = os.path.join(
        BASE_DIR,
        "data"
    )


    print("MNIST data path:", DATA_DIR)



    train_data = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transform
    )


    test_data = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform
    )


    train_size = 50000
    val_size = 10000


    train_dataset, val_dataset = random_split(
        train_data,
        [train_size, val_size]
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )


    return (
        train_loader,
        val_loader,
        test_loader
    )