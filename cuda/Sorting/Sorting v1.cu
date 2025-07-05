#include "solve.h"
#include <cub/device/device_radix_sort.cuh>

void solve(float* data, int N) {
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, data, data, N);
    
    void* d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // 升序
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, data, data, N);
    
    cudaFree(d_temp_storage);
}