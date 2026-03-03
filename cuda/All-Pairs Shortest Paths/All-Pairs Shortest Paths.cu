#include <cuda_runtime.h>

__global__ void floyd_warshall_kernel(float* output, int n, int k) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < n && col < n) {
        // dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        float c = output[row * n + col];
        float p = output[row * n + k] + output[k * n + col];
        if (p < c) {
            output[row * n + col] = p;
        }
    }     
}

// dist, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* dist, float* output, int N) {
    // N * N 矩阵拷贝到 output 里
    cudaMemcpy(output, dist, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    for (int k = 0; k < N; ++k) {
        floyd_warshall_kernel<<<grid, block>>>(output, N, k);
    }
}