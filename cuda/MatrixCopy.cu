// https://leetgpu.com/challenges/matrix-copy

#include "solve.h"
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        B[idx] = A[idx];
    }
}

// A, B are device pointers (i.e., pointers to memory on the GPU)
void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}