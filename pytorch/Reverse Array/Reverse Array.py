import torch

def solve(input: torch.Tensor, N: int):
    if N <= 1:
        return
    with torch.cuda.stream(torch.cuda.Stream()):
        indices = torch.arange(0, N//2, device='cuda', dtype=torch.long)
        
        swap_indices = N - 1 - indices
        
        left_vals = input[indices]
        input[indices] = input[swap_indices]
        input[swap_indices] = left_vals