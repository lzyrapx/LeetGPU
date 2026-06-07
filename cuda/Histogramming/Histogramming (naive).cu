#include <cuda_runtime.h>

__global__ void histogram_kernel(const int* input, int* histogram, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int value = input[idx];
    atomicAdd(&histogram[value], 1);
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    histogram_kernel<<<numBlocks, blockSize>>>(input, histogram, N);
    cudaDeviceSynchronize();
}