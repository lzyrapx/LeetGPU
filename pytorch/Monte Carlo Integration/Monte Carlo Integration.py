import torch

# y_samples, result are tensors on the GPU
def solve(y_samples: torch.Tensor, result: torch.Tensor, a: float, b: float, n_samples: int):
    # single CUDA reduction kernel
    avg = y_samples.mean()              
    result.copy_((b - a) * avg)
