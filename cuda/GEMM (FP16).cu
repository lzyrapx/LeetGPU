#include "solve.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // 计算当前线程处理的矩阵位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // 使用 float 累加防止精度丢失
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = __half2float(A[row * K + k]);
        float b = __half2float(B[k * N + col]);
        sum += a * b;
    }

    // alpha 和 beta 系数
    float c_val = (beta != 0.0f) ? __half2float(C[row * N + col]) * beta : 0.0f;
    C[row * N + col] = __float2half_rn(sum * alpha + c_val);
}

// A, B, and C are device pointers
void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    gemm_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}