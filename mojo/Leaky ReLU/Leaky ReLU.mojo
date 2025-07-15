from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn leaky_relu_kernel(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if Int32(idx) < Int32(N):
        var val = input[idx]
        output[idx] = val if val >= 0 else 0.01 * val

# input, output are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()
    
    var blocksPerGrid = ceildiv(N, threadsPerBlock)
    
    ctx.enqueue_function[leaky_relu_kernel](
        input, output, N,
        grid_dim = blocksPerGrid,
        block_dim = threadsPerBlock
    )
    
    ctx.synchronize() 