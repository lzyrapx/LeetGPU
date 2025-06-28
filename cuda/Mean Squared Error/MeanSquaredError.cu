// https://leetgpu.com/challenges/mean-squared-error

#include "solve.h"
#include <cuda_runtime.h>

const int BLOCK_SIZE = 256;

__global__ void computeSquaredDiffAndReduce(const float* predictions, const float* targets, float* partial_sums, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = 0.0f;
    if (i < N) {
        float diff = predictions[i] - targets[i];
        sdata[tid] = diff * diff;
    }
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduceSum(float* input, float* output, int numElements) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < numElements) ? input[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void computeMSE(float* sum, float* mse, int N) {
    *mse = *sum / N;
}

void solve(const float* predictions, const float* targets, float* mse, int N) {
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* d_partial_sums;
    cudaMalloc(&d_partial_sums, gridSize * sizeof(float));

    computeSquaredDiffAndReduce<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(predictions, targets, d_partial_sums, N);

    int currentSize = gridSize;
    float* currentInput = d_partial_sums;
    while (currentSize > 1) {
        gridSize = (currentSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float* d_output;
        cudaMalloc(&d_output, gridSize * sizeof(float));
        reduceSum<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(currentInput, d_output, currentSize);
        cudaFree(currentInput);
        currentInput = d_output;
        currentSize = gridSize;
    }

    computeMSE<<<1, 1>>>(currentInput, mse, N);
    cudaFree(currentInput);
}