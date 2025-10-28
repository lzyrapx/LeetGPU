import cutlass
import cutlass.cute as cute

@cute.kernel
def transpose(input, output, rows, cols):
    block_idx_col, block_idx_row, _ = cute.arch.block_idx()
    block_dim_col , block_dim_row, _ = cute.arch.block_dim()
    thread_idx_col, thread_idx_row, _ = cute.arch.thread_idx()
    row = block_idx_row * block_dim_row + thread_idx_row
    col = block_idx_col * block_dim_col + thread_idx_col
    if row < rows and col < cols:
        output[col, row] = input[row, col]

# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, rows: cute.Int32, cols: cute.Int32):
    block_dim = 32
    blocks = ((cols + block_dim - 1) // block_dim, (rows + block_dim - 1) // block_dim, 1)
    transpose(input, output, rows, cols).launch(
        grid=blocks,
        block=(block_dim, block_dim, 1)
    )
