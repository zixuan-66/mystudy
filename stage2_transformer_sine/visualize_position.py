import matplotlib.pyplot as plt
import torch

from models.positional_encoding import PositionalEncoding



pe = PositionalEncoding(
    d_model=64,
    max_len=100
)


data = pe.pe[0]


plt.imshow(
    data.numpy()
)

plt.xlabel("dimension")

plt.ylabel("position")

plt.colorbar()

plt.show()