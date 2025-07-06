from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math.math import ceildiv

# GPU kernel for matrix multiplication
fn matmul_kernel(
    A: UnsafePointer[Float32],  # M x K
    B: UnsafePointer[Float32],  # K x N
    C: UnsafePointer[Float32],  # M x N
    M: Int32, K: Int32, N: Int32
):
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x

    if Int32(row) < M and Int32(col) < N:
        var sum: Float32 = 0.0
        for k in range(K):
            var a_idx = row * K + k
            var b_idx = k * N + col
            sum += A[a_idx] * B[b_idx]
        C[row * N + col] = sum

# Exported entry function
@export
def solve(
    A: UnsafePointer[Float32],  # M x K
    B: UnsafePointer[Float32],  # K x N
    C: UnsafePointer[Float32],  # M x N
    M: Int32, K: Int32, N: Int32
):
    var BLOCK_SIZE_X: Int32 = 16
    var BLOCK_SIZE_Y: Int32 = 16

    var grid_dim_x = ceildiv(N, BLOCK_SIZE_X)
    var grid_dim_y = ceildiv(M, BLOCK_SIZE_Y)

    var ctx = DeviceContext()
    ctx.enqueue_function[matmul_kernel](
        A, B, C, M, K, N,
        grid_dim = (grid_dim_x, grid_dim_y),
        block_dim = (BLOCK_SIZE_X, BLOCK_SIZE_Y)
    )
    ctx.synchronize()