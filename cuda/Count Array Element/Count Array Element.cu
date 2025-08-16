#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    // dynamic share memory
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // total number of thread
    int stride = gridDim.x * blockDim.x;
    
    int count = 0;
    while(i < N) {
        if (input[i] == K) {
            count++;            
        }
        i += stride;
    }
    sdata[tid] = count;
    __syncthreads();

    // reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
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
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(input, output, N, K);
    cudaDeviceSynchronize();
}