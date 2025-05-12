
https://leetgpu.com/challenges/softmax-attention

#include "solve.h"
#include <cuda_runtime.h>
#include <math.h>

// 调整核函数参数，正确计算M×N的矩阵。每个线程处理Q的第i行和K的第j行的点积。
__global__ void qkt_kernel(const float* Q, const float* K, float* S, int M, int N, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < d; ++k) {
        sum += Q[i * d + k] * K[j * d + k]; // Q[i] · K[j]
    }
    sum /= sqrtf(d);
    S[i * N + j] = sum;
}

// 处理每行的N个元素，确保正确应用行方向的softmax
__global__ void softmax_kernel(const float* S, float* P, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // 计算行最大值
    float max_val = -INFINITY;
    for (int j = tid; j < N; j += num_threads) {
        max_val = fmaxf(max_val, S[row * N + j]);
    }

    // 归约求最大值
    __shared__ float shared_max[256];
    shared_max[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    float row_max = shared_max[0];

    // 计算exp和总和
    float sum_exp = 0.0f;
    for (int j = tid; j < N; j += num_threads) {
        float exp_val = expf(S[row * N + j] - row_max);
        P[row * N + j] = exp_val;
        sum_exp += exp_val;
    }

    // 归约求和
    __shared__ float shared_sum[256];
    shared_sum[tid] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float row_sum = shared_sum[0];

    // 归一化
    for (int j = tid; j < N; j += num_threads) {
        P[row * N + j] /= row_sum;
    }
}

// 遍历N次循环，确保P与V的正确矩阵乘法
__global__ void pv_kernel(const float* P, const float* V, float* output, int M, int N, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || k >= d) return;

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += P[i * N + j] * V[j * d + k];
    }
    output[i * d + k] = sum;
}

void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *S, *P;
    cudaMalloc(&S, M * N * sizeof(float));
    cudaMalloc(&P, M * N * sizeof(float));

    // 计算 QK^T / sqrt(d)
    dim3 block_qkt(16, 16);
    dim3 grid_qkt((M + 15) / 16, (N + 15) / 16);
    qkt_kernel<<<grid_qkt, block_qkt>>>(Q, K, S, M, N, d);
    cudaDeviceSynchronize();

    // 应用 softmax
    softmax_kernel<<<M, 256>>>(S, P, M, N);
    cudaDeviceSynchronize();

    // 计算 PV
    dim3 block_pv(16, 16);
    dim3 grid_pv((M + 15) / 16, (d + 15) / 16);
    pv_kernel<<<grid_pv, block_pv>>>(P, V, output, M, N, d);
    cudaDeviceSynchronize();

    cudaFree(S);
    cudaFree(P);
}