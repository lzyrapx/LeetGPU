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
    for (int k = 0; k < K; k++) {
        sum += A_b[i * K + k] * B_b[k * N + j];
    }

    C[batch * M * N + i * N + j] = sum;
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    int total_elements = BATCH * M * N;
    if (total_elements == 0) return;

    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    batched_matmul_kernel<<<blocks_per_grid, threads_per_block>>>(A, B, C, BATCH, M, N, K);
    cudaDeviceSynchronize();
}