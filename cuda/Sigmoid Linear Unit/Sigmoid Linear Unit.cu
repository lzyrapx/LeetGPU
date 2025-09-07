#include <cuda_runtime.h>

__global__ void sigmoid_kernel(float const* input, float* output, int N) {
    int const index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index < N) {
        float const x = input[index];
        output[index] = x * (1.0f / (1.0f + exp(-x)));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int const threads_per_block = 256;
    int const blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    sigmoid_kernel<<<threads_per_block, blocks_per_grid>>>(input, output, N);
    cudaDeviceSynchronize();
}
