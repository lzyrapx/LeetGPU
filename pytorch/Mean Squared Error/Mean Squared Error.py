import torch

# predictions, targets, mse are tensors on the GPU
def solve(predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
    squared_diff = (predictions - targets) ** 2
    
    sum_squared_diff = torch.sum(squared_diff)
    
    mse[0] = sum_squared_diff / N