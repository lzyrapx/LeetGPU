from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn invert_kernel(image: UnsafePointer[UInt8], width: Int32, height: Int32):
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total_pixels = width * height

    if Int32(idx) < total_pixels:
        var base = idx * 4
        image[base + 0] = 255 - image[base + 0]  # R
        image[base + 1] = 255 - image[base + 1]  # G
        image[base + 2] = 255 - image[base + 2]  # B
        # image[base + 3] is Alpha, unchanged

# image is a device pointer (i.e. pointer to memory on the GPU)
@export                         
def solve(image: UnsafePointer[UInt8], width: Int32, height: Int32):
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()
    
    var total_pixels = width * height
    var blocksPerGrid = ceildiv(total_pixels, threadsPerBlock)
    
    ctx.enqueue_function[invert_kernel](
        image, width, height,
        grid_dim = blocksPerGrid,
        block_dim = threadsPerBlock
    )
    
    ctx.synchronize() 