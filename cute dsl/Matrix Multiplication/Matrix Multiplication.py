import cutlass
import cutlass.cute as cute

@cute.kernel
def matmul_kernel(A, B, C, M, N, K):
    block_idx_col, block_idx_row, _ = cute.arch.block_idx()
    block_dim_col , block_dim_row, _ = cute.arch.block_dim()
    thread_idx_col, thread_idx_row, _ = cute.arch.thread_idx()
    row = block_idx_row * block_dim_row + thread_idx_row
    col = block_idx_col * block_dim_col + thread_idx_col

    if row < M and col < K:
        sum = 0.0
        for i in range(0, N):
            sum += A[row, i] * B[i, col]
        C[row, col] = sum

# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M: cute.Int32, N: cute.Int32, K: cute.Int32):
    block_dim = 16
    blocks = ((K + block_dim - 1) // block_dim, (M + block_dim - 1) // block_dim, 1)
    matmul_kernel(A, B, C, M, N, K).launch(
        grid=blocks,
        block=(block_dim, block_dim, 1)
    )
