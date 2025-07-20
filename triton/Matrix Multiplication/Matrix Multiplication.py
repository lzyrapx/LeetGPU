# https://leetgpu.com/challenges/matrix-multiplication

import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    m = pid_m * BLOCK_SIZE_M
    k = pid_k * BLOCK_SIZE_K

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)

    for pid_n in range(num_n_blocks):

        n = pid_n * BLOCK_SIZE_N

        offs_am = m + tl.arange(0, BLOCK_SIZE_M)
        offs_an = n + tl.arange(0, BLOCK_SIZE_N)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_an[None, :] * stride_an)

        mask_a = (offs_am[:, None] < M) & (offs_an[None, :] < N)

        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)

        offs_bn = n + tl.arange(0, BLOCK_SIZE_N)
        offs_bk = k + tl.arange(0, BLOCK_SIZE_K)

        b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk)

        mask_b = (offs_bn[:, None] < N) & (offs_bk[None, :] < K)

        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a_tile, b_tile, input_precision="ieee")

    offs_cm = m + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = k + tl.arange(0, BLOCK_SIZE_K)

    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_ck[None, :] * stride_ck)
    mask_c = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)

    tl.store(c_ptrs, acc, mask=mask_c)

def solve(a_ptr: int, b_ptr: int, c_ptr: int, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 64

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K))

    matrix_multiplication_kernel[grid](
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )