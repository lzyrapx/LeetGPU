#include <cuda_runtime.h>

__global__ void subarray_sum_kernel(const int* input, int* output, int N, int S, int E) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int index = S + blockIdx.x * blockDim.x + tid;
    int value = (index <= E) ? input[index] : 0;
    sdata[tid] = value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    cudaMemset(output, 0, sizeof(int));
    int L = E - S + 1;
    if (L <= 0) {
        return;
    }
    int threadsPerBlock = 256;
    int blocksPerGrid = (L + threadsPerBlock - 1) / threadsPerBlock;
    subarray_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, S, E);
    cudaDeviceSynchronize();
}