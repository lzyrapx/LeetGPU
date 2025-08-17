#include <cuda_runtime.h>
#include <cmath>

__global__ void compute_means_vars(const float* input, float* means, float* vars, int N, int C) {
    int j = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    extern __shared__ float sdata[];

    float sum = 0.0f;
    for (int i = tid; i < N; i += blockSize) {
        int idx = i * C + j;
        sum += input[idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    // tree reduction
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sdata[0] = sdata[0] / N;
    }
    __syncthreads();
    float mean_val = sdata[0];

    float sq_sum = 0.0f;
    for (int i = tid; i < N; i += blockSize) {
        int idx = i * C + j;
        float diff = input[idx] - mean_val;
        sq_sum += diff * diff;
    }
    sdata[tid] = sq_sum;
    __syncthreads();

    // tree reduction
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float variance = sdata[0] / N;
        means[j] = mean_val;
        vars[j] = variance;
    }
}

__global__ void batch_normalize(const float* input, const float* gamma, const float* beta, float* output,
                          const float* means, const float* vars, int N, int C, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx < total) {
        int i = idx / C;
        int j = idx % C;
        float mean_j = means[j];
        float var_j = vars[j];
        float inv_std = 1.0f / sqrtf(var_j + eps);
        float x_hat = (input[idx] - mean_j) * inv_std;
        output[idx] = gamma[j] * x_hat + beta[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, 
                     float* output, int N, int C, float eps) {
    if (N <= 0 || C <= 0) {
        return;
    }

    float *d_means, *d_vars;
    cudaMalloc((void**)&d_means, C * sizeof(float));
    cudaMalloc((void**)&d_vars, C * sizeof(float));

    int blockSize = 256;
    compute_means_vars<<<C, blockSize, blockSize * sizeof(float)>>>(input, d_means, d_vars, N, C);

    int total_elements = N * C;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    batch_normalize<<<numBlocks, blockSize>>>(input, gamma, beta, output, d_means, d_vars, N, C, eps);

    cudaDeviceSynchronize();

    cudaFree(d_means);
    cudaFree(d_vars);
}