// https://leetgpu.com/challenges/2d-convolution

#include "solve.h"
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* kernel, float* output,
                                   int input_rows, int input_cols,
                                   int kernel_rows, int kernel_cols,
                                   int output_rows, int output_cols) {
    // Calculate output coordinates for this thread
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < output_rows && j < output_cols) {
        float sum = 0.0f;
        // Iterate over kernel elements
        for (int m = 0; m < kernel_rows; ++m) {
            for (int n = 0; n < kernel_cols; ++n) {
                int input_row = i + m;
                int input_col = j + n;
                // Check input boundaries (implicitly handled by valid convolution)
                sum += input[input_row * input_cols + input_col] * kernel[m * kernel_cols + n];
            }
        }
        // Write result to output
        output[i * output_cols + j] = sum;
    }
}

void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    // Calculate output dimensions
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    // Set block and grid dimensions
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((output_cols + block.x - 1) / block.x, (output_rows + block.y - 1) / block.y);

    // Launch kernel
    convolution_kernel<<<grid, block>>>(input, kernel, output,
                                        input_rows, input_cols,
                                        kernel_rows, kernel_cols,
                                        output_rows, output_cols);
}