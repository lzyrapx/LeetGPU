#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    __shared__ int shared_count[256];  // 16x16=256 threads per block

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_in_block = threadIdx.y * blockDim.x + threadIdx.x;

    int value = 0;
    if (row < N && col < M) {
        int index = row * M + col;
        if (input[index] == K) {
            value = 1;
        }
    }
    shared_count[idx_in_block] = value;
    __syncthreads();

    // 128 = 256 / 2
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (idx_in_block < stride) {
            shared_count[idx_in_block] += shared_count[idx_in_block + stride];
        }
        __syncthreads();
    }

    if (idx_in_block == 0) {
        atomicAdd(output, shared_count[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}