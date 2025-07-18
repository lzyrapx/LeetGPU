import torch
import torch.nn as nn
import torch.nn.functional as F


# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    input_view = input.view(1, 1, -1)
    kernel_view = kernel.view(1, 1, -1)
    
    result = F.conv1d(input_view, kernel_view, padding=0)
    output.copy_(result.squeeze())