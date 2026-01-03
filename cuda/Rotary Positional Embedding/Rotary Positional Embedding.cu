#include <cuda_runtime.h>

__global__ void rope_kernel(float* Q, float* cos, float* sin, float* output, int M, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < M * D) {
        int row = idx / D; // token index (0 to M-1)
        int col = idx % D; // dimension index (0 to D-1)
        
        // the base index for this row
        int base_idx = row * D;
        
        // the current element
        float q_val = Q[base_idx + col];
        float cos_val = cos[base_idx + col];
        float sin_val = sin[base_idx + col];
        
        // calc rotate_half operation
        float rotated_val;
        int half_D = D / 2;
        
        if (col < half_D) {
            // first half: use negative of element in second half
            rotated_val = -Q[base_idx + col + half_D];
        } else {
            // second half: use positive of element in first half
            rotated_val = Q[base_idx + col - half_D];
        }
        // RoPE formula: output = Q * cos + rotate_half(Q) * sin
        output[base_idx + col] = q_val * cos_val + rotated_val * sin_val;
    }
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    int total_elements = M * D;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    rope_kernel<<<grid_size, block_size>>>(Q, cos, sin, output, M, D);
    cudaDeviceSynchronize();
}
