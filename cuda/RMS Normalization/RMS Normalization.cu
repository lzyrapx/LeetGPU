#include <cuda_runtime.h>
#include <cmath>

__global__ void partial_sum_kernel(const float* input, int N, float* partial_sums) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    float sum = 0.0f;
    while (i < N) {
        sum += input[i] * input[i];
        i += gridDim.x * blockDim.x;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void normalize_kernel(const float* input, float gamma, float beta, float rms, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float normalized = input[idx] / rms;
        output[idx] = gamma * normalized + beta;
    }
}

extern "C" void solve(const float* input, float gamma, float beta, 
                     float* output, int N, float eps) {
    const int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    
    float* d_partial_sums;
    cudaMalloc(&d_partial_sums, blocks * sizeof(float));
    
    partial_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(input, N, d_partial_sums);
    
    float* h_partial_sums = new float[blocks];
    cudaMemcpy(h_partial_sums, d_partial_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_sum = 0.0f;
    for (int i = 0; i < blocks; i++) {
        total_sum += h_partial_sums[i];
    }
    
    float rms = sqrtf(total_sum / N + eps);
    
    normalize_kernel<<<blocks, threads>>>(input, gamma, beta, rms, output, N);
    
    cudaDeviceSynchronize();
    
    delete[] h_partial_sums;
    cudaFree(d_partial_sums);
}