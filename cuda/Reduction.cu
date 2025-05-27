// https://leetgpu.com/challenges/reduction

#include "solve.h"
#include <cuda_runtime.h>

// 并行归约求和
__global__ void reduction_kernel(const float* input, float* output, int N) {
    extern __shared__ float s_data[];  // share memory
    
    // 每个线程加载两个全局内存元素到共享内存
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;
    s_data[tid] = (i < N) ? input[i] : 0;
    if (i + blockDim.x < N) {
        s_data[tid] += input[i + blockDim.x];
    }

    // 块内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
    }

    // share memory => global memory
    if (tid == 0) {
        atomicAdd(output, s_data[0]);
    }
}

void solve(const float* input, float* output, int N) {
    int threads = 1024;
    int blocks = (N + 2 * threads - 1) / (2 * threads);
    reduction_kernel<<<blocks, threads, threads*sizeof(float)>>>(input, output, N);
    cudaDeviceSynchronize();
}