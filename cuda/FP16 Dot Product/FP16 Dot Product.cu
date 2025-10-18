#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

__global__ void partial_dot_kernel(const half* A, const half* B, float* partial_sums, int N) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int elements_per_block = min(blockDim.x, N - block_start);
    
    float local_sum = 0.0f;
    
    for (int i = tid; i < elements_per_block; i += blockDim.x) {
        int global_idx = block_start + i;
        if (global_idx < N) {
            float a_val = __half2float(A[global_idx]);
            float b_val = __half2float(B[global_idx]);
            local_sum += a_val * b_val;
        }
    }
    
    shared_mem[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_mem[0];
    }
}

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {
    int block_size = 1024;
    int num_blocks = (N + block_size - 1) / block_size;
    
    float* d_partial_sums;
    float* h_partial_sums = new float[num_blocks];
    cudaMalloc(&d_partial_sums, num_blocks * sizeof(float));

    size_t shared_mem_size = block_size * sizeof(float);
    partial_dot_kernel<<<num_blocks, block_size, shared_mem_size>>>(A, B, d_partial_sums, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_partial_sums, d_partial_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float final_sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        final_sum += h_partial_sums[i];
    }

    half final_result = __float2half(final_sum);
    cudaMemcpy(result, &final_result, sizeof(half), cudaMemcpyHostToDevice);
    
    cudaFree(d_partial_sums);
    delete[] h_partial_sums;
}