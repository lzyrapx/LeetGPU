// https://leetgpu.com/challenges/monte-carlo-integration

#include "solve.h"
#include <cuda_runtime.h>

// 计算样本总和（使用并行归约）
__global__ void sum_kernel(const float* y, float* total_sum, int n) {
    // 动态分配共享内存，用于块内归约
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;  // 网格中所有线程的总数
    float thread_sum = 0.0f;

    // 每个线程处理多个元素（网格跨步循环）
    for (int idx = global_idx; idx < n; idx += stride) {
        thread_sum += y[idx];
    }

    // 将线程局部和存入共享内存
    sdata[tid] = thread_sum;
    __syncthreads();

    // 并行归约：树形求和（要求blockDim.x是2的幂）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 块内0号线程将部分和原子累加到全局总和
    if (tid == 0) {
        atomicAdd(total_sum, sdata[0]);
    }
}

// 计算最终积分结果
__global__ void result_kernel(float* result, const float* total_sum, float interval, int n_samples) {
    // 计算样本平均值
    float avg = *total_sum / n_samples;
    // 计算积分值: (b - a) * 平均值
    *result = interval * avg;
}

void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    float interval = b - a;  // 积分区间长度
    
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    const int blockSize = 256;
    int gridSize = (n_samples + blockSize - 1) / blockSize;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (gridSize > prop.maxGridSize[0]) {
        gridSize = prop.maxGridSize[0];
    }

    size_t sharedMemSize = blockSize * sizeof(float);  // 共享内存大小
    sum_kernel<<<gridSize, blockSize, sharedMemSize>>>(y_samples, d_sum, n_samples);
    cudaDeviceSynchronize();

    result_kernel<<<1, 1>>>(result, d_sum, interval, n_samples);
    cudaDeviceSynchronize();
    
    cudaFree(d_sum);
}