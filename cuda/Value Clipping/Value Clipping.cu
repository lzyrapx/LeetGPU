#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float low, float high, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        float tmp = input[idx] > high ? high : input[idx];
        output[idx] = tmp < low ? low : tmp;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
