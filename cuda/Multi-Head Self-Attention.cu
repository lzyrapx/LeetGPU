// https://leetgpu.com/challenges/multi-head-self-attention

#include "solve.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void compute_A(float* d_A, const float* Q, const float* K, int N, int d_model, int h, int d_k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = h * N * N;
    if (tid >= total) return;

    int i = tid / (N * N);
    int remainder = tid % (N * N);
    int n = remainder / N;
    int m = remainder % N;

    float sum = 0.0f;
    for (int k = 0; k < d_k; ++k) {
        int q_idx = n * d_model + i * d_k + k;
        int k_idx = m * d_model + i * d_k + k;
        sum += Q[q_idx] * K[k_idx];
    }
    sum /= sqrtf(static_cast<float>(d_k));
    d_A[tid] = sum;
}

__global__ void compute_softmax(float* d_S, const float* d_A, int N, int h) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = h * N;
    if (tid >= total_rows) return;

    int i = tid / N;
    int n = tid % N;

    // Compute max value for numerical stability
    float max_val = -INFINITY;
    for (int m = 0; m < N; ++m) {
        int a_idx = i * N * N + n * N + m;
        max_val = fmaxf(max_val, d_A[a_idx]);
    }

    // Compute exponentials and sum
    float sum = 0.0f;
    for (int m = 0; m < N; ++m) {
        int a_idx = i * N * N + n * N + m;
        sum += expf(d_A[a_idx] - max_val);
    }

    // Compute softmax and write to d_S
    for (int m = 0; m < N; ++m) {
        int a_idx = i * N * N + n * N + m;
        d_S[a_idx] = expf(d_A[a_idx] - max_val) / sum;
    }
}

__global__ void compute_O(float* output, const float* d_S, const float* V, int N, int d_model, int h, int d_k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = h * N * d_k;
    if (tid >= total) return;

    int i = tid / (N * d_k);
    int remainder = tid % (N * d_k);
    int n = remainder / d_k;
    int j = remainder % d_k;

    float sum = 0.0f;
    for (int m = 0; m < N; ++m) {
        int s_idx = i * N * N + n * N + m;
        int v_idx = m * d_model + i * d_k + j;
        sum += d_S[s_idx] * V[v_idx];
    }

    int out_col = i * d_k + j;
    output[n * d_model + out_col] = sum;
}

void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
    int d_k = d_model / h;

    float *d_A, *d_S;
    cudaMalloc(&d_A, h * N * N * sizeof(float));
    cudaMalloc(&d_S, h * N * N * sizeof(float));

    // Compute A = Q*K^T / sqrt(d_k)
    int total_A = h * N * N;
    int block_size = 256;
    int grid_size = (total_A + block_size - 1) / block_size;
    compute_A<<<grid_size, block_size>>>(d_A, Q, K, N, d_model, h, d_k);

    // Compute softmax(A)
    int total_softmax = h * N;
    grid_size = (total_softmax + block_size - 1) / block_size;
    compute_softmax<<<grid_size, block_size>>>(d_S, d_A, N, h);

    // Compute output = softmax(A)*V and concatenate
    int total_O = h * N * d_k;
    grid_size = (total_O + block_size - 1) / block_size;
    compute_O<<<grid_size, block_size>>>(output, d_S, V, N, d_model, h, d_k);

    cudaFree(d_A);
    cudaFree(d_S);
}