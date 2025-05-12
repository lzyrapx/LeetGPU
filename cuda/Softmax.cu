// https://leetgpu.com/challenges/softmax

#include "solve.h"
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // naive softmax
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // 每个线程遍历整个数组找到最大值（简单但低效）
    float max_val = input[0];
    for (int i = 1; i < N; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }

    // 计算当前元素的指数（减去最大值）
    float exp_val = expf(input[idx] - max_val);
    output[idx] = exp_val;

    // 确保全局内存写入完成
    __threadfence();

    // 计算所有指数值的总和
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) sum += output[i];

    // 归一化得到最终结果
    output[idx] = exp_val / sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}