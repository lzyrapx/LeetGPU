#include <cuda_runtime.h>

__global__ void count_3d_equal_kernel(const int* input, int* output, int N, int M, int K, int P) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    int count = 0;
    if (i < N && j < M && k < K) {
        // compute the linear index
        int idx = i * (M * K) + j * K + k;
        if (input[idx] == P) {
            count = 1;
        }
    }

    // shared memory for reduction (fixed size 512 for 8x8x8 block)
    __shared__ int sdata[512];

    // flatten the thread index in the block
    int tid = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    sdata[tid] = count;
    __syncthreads();

    // tree reduction
    // 256 = 512 / 2
    for (int s = 256; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // share memory => global memory
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) {
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                      (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    count_3d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K, P);
    cudaDeviceSynchronize();
}