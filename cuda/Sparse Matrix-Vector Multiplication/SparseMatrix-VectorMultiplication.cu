#include <cuda_runtime.h>

// sparse-matrix-vector-multiplication kernel
__global__ void spmv_kernel(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M) return;

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float element = A[row * N + i];
        if (element != 0.0f) sum += element * x[i];
    }
    y[row] = sum;
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int threads_per_block = 256;
    int blocks_per_grid = (M + threads_per_block - 1) / threads_per_block;
    spmv_kernel<<<blocks_per_grid, threads_per_block>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}