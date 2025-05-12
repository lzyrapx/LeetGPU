// https://leetgpu.com/challenges/top-k-selection

#include "solve.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// input, output are device pointers
void solve(const float* input, float* output, int N, int k) {
    float *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    float *d_sorted;
    cudaMalloc(&d_sorted, N * sizeof(float));

    cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, input, d_sorted, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, input, d_sorted, N);
    cudaMemcpy(output, d_sorted, k * sizeof(float), cudaMemcpyDeviceToDevice);
    
    cudaFree(d_sorted);
    cudaFree(d_temp_storage);
    cudaDeviceSynchronize();
}