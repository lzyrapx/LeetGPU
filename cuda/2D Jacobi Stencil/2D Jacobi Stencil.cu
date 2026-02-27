#include <cuda_runtime.h>

__global__ void two_dim_jacobi_kernel(const float* input, float* output, int rows, int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col >= cols || row >= rows) return;
    if (col == cols - 1 || col == 0 || row == rows - 1 || row == 0) {
        output[row * cols + col] = input[row * cols + col];
    } else {
        int idx = row * cols + col;
        int i1 = (row - 1) * cols + col, i2 = (row + 1) * cols + col;
        int i3 = row * cols + (col - 1), i4 = row * cols + (col + 1);
        output[idx] = 0.25f * (input[i1] + input[i2] + input[i3] + input[i4]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 16 * 16 = 256 threads
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    two_dim_jacobi_kernel<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}