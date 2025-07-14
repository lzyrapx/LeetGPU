from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn reverse_array_kernel(input: UnsafePointer[Float32], N: Int32):
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var mid = N // 2
    if Int32(idx) < Int32(mid):
        var opp_idx = N - 1 - idx
        var temp = input[idx]
        input[idx] = input[opp_idx]
        input[opp_idx] = temp

# input is a device pointer (i.e. pointer to memory on the GPU)
@export                         
def solve(input: UnsafePointer[Float32], N: Int32):
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()
    
    var blocksPerGrid = ceildiv(N, threadsPerBlock)
    
    ctx.enqueue_function[reverse_array_kernel](
        input, N,
        grid_dim = blocksPerGrid,
        block_dim = threadsPerBlock
    )
    
    ctx.synchronize() 