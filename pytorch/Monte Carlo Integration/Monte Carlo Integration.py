import torch

# y_samples, result are tensors on the GPU
def solve(y_samples: torch.Tensor, result: torch.Tensor, a: float, b: float, n_samples: int):
    result[:] = y_samples.sum() / n_samples * (b - a)