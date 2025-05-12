// https://leetgpu.com/challenges/batched-matrix-multiplication-fp32

#include "solve.h"
#include <cuda_runtime.h>

__global__ void batched_matmul_kernel(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = BATCH * M * N;
    if (global_idx >= total_elements) return;

    int batch = global_idx / (M * N);
    int element_in_batch = global_idx % (M * N);
    int i = element_in_batch / N;
    int j = element_in_batch % N;

    const float* A_b = A + batch * M * K;
    const float* B_b = B + batch * K * N;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A_b[i * K + k] * B_b[k * N + j];
    }

    C[batch * M * N + i * N + j] = sum;
}

void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    int totalElements = BATCH * M * N;
    if (totalElements == 0) return;

    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    batched_matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, BATCH, M, N, K);
    cudaDeviceSynchronize();
}