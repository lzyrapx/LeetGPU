#include <cuda_runtime.h>

__global__ void sum_kernel(const int* input, int* output, int M, int K, int S_DEP, 
                        int S_ROW, int S_COL, int layer_size, int col_size, int total) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int val = 0;
    if (idx < total) {
        int dep_index = idx / layer_size;
        int rem = idx % layer_size;
        int row_index = rem / col_size;
        int col_index = rem % col_size;

        int actual_dep = S_DEP + dep_index;
        int actual_row = S_ROW + row_index;
        int actual_col = S_COL + col_index;

        val = input[actual_dep * (M * K) + actual_row * K + actual_col];
    }

    s_data[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, s_data[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int dep_size = E_DEP - S_DEP + 1;
    int row_size = E_ROW - S_ROW + 1;
    int col_size = E_COL - S_COL + 1;
    int layer_size = row_size * col_size;
    int total = dep_size * row_size * col_size;

    if (total == 0) {
        cudaMemset(output, 0, sizeof(int));
        return;
    }

    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    cudaMemset(output, 0, sizeof(int));
    sum_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(int)>>>(input, output, M, K, S_DEP, S_ROW, S_COL, layer_size, col_size, total);
    cudaDeviceSynchronize();
}