// https://leetgpu.com/challenges/histogramming

#include "solve.h"
#include <cuda_runtime.h>

__global__ void computeHistogramKernel(const int* input, int* histogram, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int value = input[idx];
        atomicAdd(&histogram[value], 1);
    }
}

void solve(const int* input, int* histogram, int N, int num_bins) {
    // Initialize histogram to zero
    cudaMemset(histogram, 0, num_bins * sizeof(int));

    // Configure kernel launch parameters
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Launch kernel to compute histogram
    computeHistogramKernel<<<numBlocks, blockSize>>>(input, histogram, N);

    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
}

#include "solve.h"
#include <cuda_runtime.h>

#define LOAD_FACTOR 4  // 每个线程处理4个元素

__global__ void computeHistogramKernel(const int* input, int* histogram, int N, int num_bins) {
    extern __shared__ int s_hist[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    const int block_size = blockDim.x * LOAD_FACTOR;
    const int total_threads = gridDim.x * block_size;

    // 初始化共享内存直方图（细粒度初始化）
    for (int i = tid; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // 逐元素加载（避免向量化强制转换）
    for (int pos = bid * block_size + tid; pos < N; pos += total_threads) {
        int vals[LOAD_FACTOR];
        
        // 手动加载4个元素（避免int4强制对齐）
        #pragma unroll
        for (int k = 0; k < LOAD_FACTOR; k++) {
            const int idx = pos + k * blockDim.x;
            vals[k] = (idx < N) ? input[idx] : -1;  // -1作为无效标记
        }

        // 更新共享内存直方图
        #pragma unroll
        for (int k = 0; k < LOAD_FACTOR; k++) {
            if (vals[k] >= 0 && vals[k] < num_bins) {  // 双重边界检查
                atomicAdd(&s_hist[vals[k]], 1);
            }
        }
    }
    __syncthreads();

    // 合并到全局直方图（交错存储）
    for (int i = tid; i < num_bins; i += blockDim.x) {
        if (s_hist[i] > 0) {
            atomicAdd(&histogram[i], s_hist[i]);
        }
    }
}

void solve(const int* input, int* histogram, int N, int num_bins) {
    cudaMemset(histogram, 0, num_bins * sizeof(int));
    const int block_size = 256;
    int num_blocks;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_blocks = prop.maxGridSize[0];
    num_blocks = min(max_blocks, (N + block_size * LOAD_FACTOR - 1) / (block_size * LOAD_FACTOR));

    size_t shared_mem = num_bins * sizeof(int);
    computeHistogramKernel<<<num_blocks, block_size, shared_mem>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();

}