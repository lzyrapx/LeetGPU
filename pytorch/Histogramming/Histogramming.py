import torch

# input, histogram are tensors on the GPU
def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    histogram.copy_(torch.bincount(input, minlength=num_bins)) 