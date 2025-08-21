import torch

# A, x, y are tensors on the GPU
def solve(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, N: int, nnz: int):
    # Reshape A to a 2D matrix
    A_matrix = A.view(M, N)
    result = torch.matmul(A_matrix, x)
    y.copy_(result) 