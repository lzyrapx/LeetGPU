from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

alias Float4 = SIMD[DType.float32, 4]

fn vector_add_kernel(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], N: Int32):
    var idx = thread_idx.x + block_dim.x * block_idx.x
    var start_float_idx = idx * 4

    if start_float_idx + 3 < UInt(N):
        var a = A.load[width=4](start_float_idx)
        var b = B.load[width=4](start_float_idx)
        var c = a + b
        C.store[width=4](start_float_idx, c)
    elif start_float_idx < UInt(N):
        for i in range(4):
            var current_float_idx = start_float_idx + i
            if current_float_idx < UInt(N):       
                C[current_float_idx] = A[current_float_idx] + B[current_float_idx]
            else:
                break
    

# A, B, C are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], N: Int32):
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(N, BLOCK_SIZE * 4)

    ctx.enqueue_function[vector_add_kernel](
        A, B, C, N,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()