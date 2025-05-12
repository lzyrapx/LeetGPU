// https://leetgpu.com/challenges/categorical-cross-entropy-loss

#include "solve.h"
#include <cuda_runtime.h>

__global__ void cross_entropy_kernel(const float* logits, const int* true_labels, float* loss_array, int N, int C) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    const float* z_j = logits + j * C;
    int y_j = true_labels[j];

    // Compute max logit for numerical stability
    float max_z = z_j[0];
    for (int k = 1; k < C; ++k) {
        if (z_j[k] > max_z) {
            max_z = z_j[k];
        }
    }

    // Compute sum of exponentials
    float sum_exp = 0.0f;
    for (int k = 0; k < C; ++k) {
        sum_exp += expf(z_j[k] - max_z);
    }

    // Compute log-sum-exp and loss
    float log_sum_exp = logf(sum_exp) + max_z;
    float loss_j = log_sum_exp - z_j[y_j];
    loss_array[j] = loss_j;
}

__global__ void sum_kernel(const float* input, float* sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(sum, input[idx]);
    }
}

void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    float* d_loss_array;
    cudaMalloc(&d_loss_array, N * sizeof(float));

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    cross_entropy_kernel<<<grid_size, block_size>>>(logits, true_labels, d_loss_array, N, C);

    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    sum_kernel<<<grid_size, block_size>>>(d_loss_array, d_sum, N);

    float h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float avg_loss = h_sum / N;

    cudaMemcpy(loss, &avg_loss, sizeof(float), cudaMemcpyHostToDevice);

    cudaFree(d_loss_array);
    cudaFree(d_sum);
}