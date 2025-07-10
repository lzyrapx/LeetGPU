import tinygrad

# A, B, C are tensors on the GPU
def solve(A: tinygrad.Tensor, B: tinygrad.Tensor, C: tinygrad.Tensor, M: int, N: int, K: int):
    C.replace(A.matmul(B))