#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void batched_mat_mul_kernel(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= BATCH || row >= M || col >= N) return;
    
    // fp32 for accumulation
    float sum = 0.0f;
    
    for (int k = 0; k < K; k++) {
        // fp16 => fp32
        float a_val = __half2float(A[batch * M * K + row * K + k]);
        float b_val = __half2float(B[batch * K * N + k * N + col]);
        sum += a_val * b_val;
    }
    
    // fp32 => fp16
    C[batch * M * N + row * N + col] = __float2half(sum);
}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {

    // 16x16 thread blocks
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,  // ceil(N/16)
        (M + blockDim.y - 1) / blockDim.y,  // ceil(M/16)
        BATCH                               // one block per batch
    );
    
    batched_mat_mul_kernel<<<gridDim, blockDim>>>(A, B, C, BATCH, M, N, K);
    cudaDeviceSynchronize();
}