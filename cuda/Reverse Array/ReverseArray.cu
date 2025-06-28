// https://leetgpu.com/challenges/reverse-array

#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N / 2) return;

    int opposite = N - 1 - tid;
    float temp = input[tid];
    input[tid] = input[opposite];
    input[opposite] = temp;
}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}