from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn relu_kernel(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    var i =  block_idx.x * block_dim.x + thread_idx.x
    if Int32(i) < Int32(N):
        var val = input[i]
        output[i] = val if val >= 0 else 0

# input, output are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()
    
    var blocksPerGrid = ceildiv(N, threadsPerBlock)
    
    ctx.enqueue_function[relu_kernel](
        input, output, N,
        grid_dim = blocksPerGrid,
        block_dim = threadsPerBlock
    )
    
    ctx.synchronize() 