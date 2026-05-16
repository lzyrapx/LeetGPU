#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < K) {
        float value = 0.0f;
        // C[i][j] = \sum_{k=0}^{N-1} A[i][k] * B[k][j]
        // A 是一个 M x N 的矩阵，B 是一个 N x K 的矩阵，C 是一个 M x K 的矩阵
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * K + col];
        }
        C[row * K + col] = value;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
