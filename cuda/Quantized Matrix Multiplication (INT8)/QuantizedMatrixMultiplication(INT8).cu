// https://leetgpu.com/challenges/quantized-matrix-multiplication-int8

#include "solve.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void quantized_matmul_kernel(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, 
                                       float scale_A, float scale_B, float scale_C, 
                                       int zero_point_A, int zero_point_B, int zero_point_C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
        int8_t a_val = A[row * K + k];
        int8_t b_val = B[k * N + col];
        sum += (static_cast<int32_t>(a_val) - zero_point_A) * 
               (static_cast<int32_t>(b_val) - zero_point_B);
    }

    float scale_factor = (scale_A * scale_B) / scale_C;
    float scaled_value = static_cast<float>(sum) * scale_factor;
    int32_t rounded = static_cast<int32_t>(roundf(scaled_value)) + zero_point_C;

    rounded = max(-128, min(127, rounded));
    C[row * N + col] = static_cast<int8_t>(rounded);
}

// A, B, C are device pointers
void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, 
        float scale_A, float scale_B, float scale_C,
        int zero_point_A, int zero_point_B, int zero_point_C) {
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    quantized_matmul_kernel<<<grid, block>>>(A, B, C, M, N, K, scale_A, scale_B, scale_C, 
                                            zero_point_A, zero_point_B, zero_point_C);
    cudaDeviceSynchronize();
} 