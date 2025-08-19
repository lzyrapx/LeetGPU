import torch

# A, B, result are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    torch.dot(A, B, out=result)