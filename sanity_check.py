import torch

t = torch.load("data/tensors/0X4B7C480E6B6C5F4C.pt")
print(t.shape)
print(t.dtype)
print(t.min(), t.max())
