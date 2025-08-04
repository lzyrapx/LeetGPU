import torch

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    x = input - input.max()
    x = torch.exp(x)
    x = x / x.sum()
    output.copy_(x)