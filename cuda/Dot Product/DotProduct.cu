// https://leetgpu.com/challenges/dot-product

#include "solve.h"
#include <cuda_runtime.h>

__global__ void dotProductKernel(const float* A, const float* B, float* tmp, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float sum = (i < N) ? A[i] * B[i] : 0.0f;
    sdata[tid] = sum;
    __syncthreads();

    // Block-wise reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        tmp[blockIdx.x] = sdata[0];
    }
}

__global__ void sumKernel(const float* tmp, float* result, int numBlocks) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;

    // Accumulate all block sums
    for (int i = tid; i < numBlocks; i += blockDim.x) {
        sdata[tid] += tmp[i];
    }
    __syncthreads();

    // Final reduction within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = sdata[0];
    }
}

// A, B, result are device pointers
void solve(const float* A, const float* B, float* result, int N) {
    const int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* tmp;
    cudaMalloc(&tmp, numBlocks * sizeof(float));

    // Launch first kernel to compute partial sums
    dotProductKernel<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(A, B, tmp, N);

    // Launch second kernel to sum all partial sums
    const int sumThreads = 256;
    sumKernel<<<1, sumThreads, sumThreads * sizeof(float)>>>(tmp, result, numBlocks);

    cudaDeviceSynchronize();
    cudaFree(tmp);
}