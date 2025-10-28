import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = bidx * bdim + tidx
    if thread_idx < N:
        a_val = A[thread_idx]
        b_val = B[thread_idx]
        C[thread_idx] = a_val + b_val

# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    kernel(A, B, C, N).launch(
        grid=(cute.ceil_div(N, 64), 1, 1),
        block=(64, 1, 1)
    )
    return C