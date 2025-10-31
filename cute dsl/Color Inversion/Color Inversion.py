import cutlass
import cutlass.cute as cute

@cute.kernel
def color_inversion(image, width, height):
    block_idx_col, block_idx_row, _ = cute.arch.block_idx()
    block_dim_col , block_dim_row, _ = cute.arch.block_dim()
    thread_idx_col, thread_idx_row, _ = cute.arch.thread_idx()
    row = block_idx_row * block_dim_row + thread_idx_row
    col = block_idx_col * block_dim_col + thread_idx_col
    
    pos = (col + row * width) * 4
    if pos < width * height * 4:
        image[pos] = cute.Uint8(255)-image[pos]
        image[pos+1] = cute.Uint8(255)-image[pos+1]
        image[pos+2] = cute.Uint8(255)-image[pos+2]

# image are tensors on the GPU
@cute.jit
def solve(image: cute.Tensor, width: cute.Int32, height: cute.Int32):
    block_dim = 16
    blocks = ((width + block_dim - 1) // block_dim, (height + block_dim - 1) // block_dim, 1)
    color_inversion(image, width, height).launch(
        grid=blocks,
        block=(block_dim, block_dim, 1)
    )