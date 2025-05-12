// https://leetgpu.com/challenges/sparse-matrix-vector-multiplication

#include "solve.h"
#include <cuda_runtime.h>

__global__ void spmv_kernel(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float element = A[row * N + j];
            if (element != 0.0f) {
                sum += element * x[j];
            }
        }
        y[row] = sum;
    }
}

void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    const int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;
    spmv_kernel<<<gridSize, blockSize>>>(A, x, y, M, N);
    cudaDeviceSynchronize(); // Ensure kernel completes
}