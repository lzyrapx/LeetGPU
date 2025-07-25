#include <cuda_runtime.h>
#include <cub/cub.cuh>

// input, output are device pointers
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, 
        temp_storage_bytes, 
        input, 
        output, 
        N
    );
    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, 
        temp_storage_bytes, 
        input, 
        output, 
        N,
        0,      // begin_bit
        32      // end_bit
    );
    
    cudaFree(d_temp_storage);       
}