import torch
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    input_ptr,      # pointer to input array (float32)
    kernel_ptr,     # pointer to kernel array (float32)
    output_ptr,     # pointer to output array (float32)
    input_size,     # total length of input
    kernel_size,    # total length of kernel
    BLOCK_SIZE: tl.constexpr  # number of outputs per program
):
    # reinterpret the raw pointers as float32 pointers
    input_ptr  = input_ptr.to(tl.pointer_type(tl.float32))
    kernel_ptr = kernel_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))

    # which block of outputs are we computing?
    pid = tl.program_id(axis=0)
    # offsets of all threads in this block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # valid output range
    max_o = input_size - kernel_size + 1
    mask  = offs < max_o

    # accumulator for each output lane
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # slide the kernel over the window
    for k in range(0, kernel_size):
        # load input[offs + k] with boundary masking
        x = tl.load(input_ptr + offs + k,
                    mask=mask,
                    other=0.0,
                    cache_modifier='.ca')
        # load a single kernel value
        w = tl.load(kernel_ptr + k)
        acc += x * w

    # write back
    tl.store(output_ptr + offs,
             acc,
             mask=mask,
             cache_modifier='.wb')


# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)
    
    conv1d_kernel[grid](
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE
    )