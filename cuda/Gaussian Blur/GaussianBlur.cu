// https://leetgpu.com/challenges/gaussian-blur

#include "solve.h"
#include <cuda_runtime.h>

__global__ void convolutionKernel(const float* input, const float* kernel, float* output,
                                  int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input_rows || j >= input_cols) {
        return;
    }
    int kh_half = kernel_rows / 2;
    int kw_half = kernel_cols / 2;
    float sum = 0.0f;
    for (int m = 0; m < kernel_rows; ++m) {
        for (int n = 0; n < kernel_cols; ++n) {
            int input_i = i + (m - kh_half);
            int input_j = j + (n - kw_half);
            float val = 0.0f;
            if (input_i >= 0 && input_i < input_rows && input_j >= 0 && input_j < input_cols) {
                val = input[input_i * input_cols + input_j];
            }
            sum += val * kernel[m * kernel_cols + n];
        }
    }
    output[i * input_cols + j] = sum;
}

// input, kernel, output are device pointers
void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    dim3 block(16, 16);
    dim3 grid((input_cols + block.x - 1) / block.x, (input_rows + block.y - 1) / block.y);

    convolutionKernel<<<grid, block>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
