import torch

# data is a tensor on the GPU
def solve(data: torch.Tensor, N: int):
    data.copy_(torch.sort(data)[0]) 