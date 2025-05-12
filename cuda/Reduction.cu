// https://leetgpu.com/challenges/reduction

#include "solve.h"
#include <cuda_runtime.h>

__global__ void reductionKernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void solve(const float* input, float* output, int N) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMemSize = threadsPerBlock * sizeof(float);

    // 分配设备内存
    float *d_input, *d_partial;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_partial, blocksPerGrid * sizeof(float));

    // 拷贝输入数据到设备
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数进行部分归约
    reductionKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_partial, N);

    // 如果部分结果数量超过一个块，需要再次调用核函数
    if (blocksPerGrid > 1) {
        float *d_final;
        cudaMalloc(&d_final, sizeof(float));
        reductionKernel<<<1, threadsPerBlock, sharedMemSize>>>(d_partial, d_final, blocksPerGrid);
        cudaMemcpy(output, d_final, sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(d_final);
    } else {
        cudaMemcpy(output, d_partial, sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_partial);
}