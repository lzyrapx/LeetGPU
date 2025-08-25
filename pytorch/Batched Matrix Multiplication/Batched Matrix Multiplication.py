import torch

# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, BATCH: int, M: int, N: int, K: int):
    A = A.view(BATCH, M, K)
    B = B.view(BATCH, K, N)
    C = C.view(BATCH, M, N)
    C.copy_(torch.bmm(A, B))