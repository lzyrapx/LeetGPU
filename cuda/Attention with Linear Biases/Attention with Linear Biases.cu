#include <cuda_runtime.h>
#include <cmath>

__global__ void attention_scores_kernel(const float* Q, const float* K, 
                                    float* attn_scores, int M, int N, int d,
                                    float inv_sqrt_d, float alpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < d; k++) {
            sum += Q[row * d + k] * K[col * d + k];
        }
        sum *= inv_sqrt_d;
        sum += alpha * (row - col);
        attn_scores[row * N + col] = sum;
    }
}

__global__ void softmax_kernel(float* attn_scores, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;

    float max_val = -INFINITY;
    for (int i = tid; i < N; i += step) {
        float val = attn_scores[row * N + i];
        if (val > max_val) max_val = val;
    }

    __shared__ float s_max[256];
    s_max[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid] < s_max[tid + s]) {
                s_max[tid] = s_max[tid + s];
            }
        }
        __syncthreads();
    }
    max_val = s_max[0];

    float sum_exp = 0.0f;
    for (int i = tid; i < N; i += step) {
        sum_exp += expf(attn_scores[row * N + i] - max_val);
    }

    __shared__ float s_sum[256];
    s_sum[tid] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    sum_exp = s_sum[0];

    for (int i = tid; i < N; i += step) {
        attn_scores[row * N + i] = expf(attn_scores[row * N + i] - max_val) / sum_exp;
    }
}

__global__ void output_kernel(const float* attn_weights, const float* V,
                        float* output, int M, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < d) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += attn_weights[row * N + k] * V[k * d + col];
        }
        output[row * d + col] = sum;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d, float alpha) {
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(d));
    float* attn_scores;
    cudaMalloc(&attn_scores, M * N * sizeof(float));

    dim3 block1(16, 16);
    dim3 grid1((N + 15) / 16, (M + 15) / 16);
    attention_scores_kernel<<<grid1, block1>>>(Q, K, attn_scores, M, N, d, inv_sqrt_d, alpha);

    int blockSize = 256;
    softmax_kernel<<<M, blockSize>>>(attn_scores, M, N);

    dim3 block3(16, 16);
    dim3 grid3((d + 15) / 16, (M + 15) / 16);
    output_kernel<<<grid3, block3>>>(attn_scores, V, output, M, N, d);

    cudaFree(attn_scores);
}