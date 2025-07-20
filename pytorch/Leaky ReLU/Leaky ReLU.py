import torch

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(torch.where(input >= 0, input, 0.01 * input))