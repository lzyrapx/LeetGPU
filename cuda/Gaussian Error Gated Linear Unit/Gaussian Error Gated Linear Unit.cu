#include <cuda_runtime.h>

// 1/sqrt(2)
#define INV_SQRT_2 0.70710678118654752440f

__global__ void geglu_kernel(const float* __restrict__ input, float* __restrict__ output, int halfN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < halfN) {
        // 读取 x1 和 x2
        // x1 对应前半部分 input[idx]
        // x2 对应后半部分 input[idx + halfN]
        // 由于 idx 是连续的，这里两次读取都是合并访存 (Coalesced Access)
        float x1 = input[idx];
        float x2 = input[idx + halfN];

        // 计算 GELU(x2)
        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        // 使用 erff (单精度误差函数)
        float arg = x2 * INV_SQRT_2;
        float erf_val = erff(arg);
        float gelu_x2 = 0.5f * x2 * (1.0f + erf_val);

        // 计算 GEGLU 并写入输出
        // GEGLU = x1 * GELU(x2)
        // 写入也是合并访存
        output[idx] = x1 * gelu_x2;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
