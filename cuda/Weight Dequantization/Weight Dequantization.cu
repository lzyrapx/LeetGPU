#include <cuda_runtime.h>

__global__ void dequantize_kernel(const int M, const int N, const int TILE_SIZE, const float* X, const float* S, float* Y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int tile_row = row / TILE_SIZE;
        int tile_col = col / TILE_SIZE;

        // s_cols = ceil(N / TILE_SIZE)
        int s_cols =  (N + TILE_SIZE - 1) / TILE_SIZE;

        float scale = S[tile_row * s_cols + tile_col];

        float val_x = X[row * N + col];

        // Y[i, j] =  val_x * scale = X[i, j] * S[scale_i, scale_j]
        Y[row * N + col] = val_x * scale;
    }
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dequantize_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, TILE_SIZE, X, S, Y);
    cudaDeviceSynchronize();
}