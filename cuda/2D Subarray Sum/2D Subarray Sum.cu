#include <cuda_runtime.h>

__global__ void row_sum_kernel(const int* input, int* row_sums, int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int row = S_ROW + blockIdx.x;
    int numCols = E_COL - S_COL + 1;
    int segment_size = (numCols + blockDim.x - 1) / blockDim.x;
    int start_col = S_COL + threadIdx.x * segment_size;
    int end_col = min(start_col + segment_size - 1, E_COL);

    int sum = 0;
    for (int col = start_col; col <= end_col; col++) {
        sum += input[row * M + col];
    }

    extern __shared__ int shared[];
    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        row_sums[blockIdx.x] = shared[0];
    }
}

__global__ void total_sum_kernel(const int* row_sums, int* output, int numRows) {
    __shared__ int sdata[1024];
    int tid = threadIdx.x;
    int total = 0;
    for (int i = tid; i < numRows; i += blockDim.x) {
        total += row_sums[i];
    }
    sdata[tid] = total;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0];
    }
}

extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int numRows = E_ROW - S_ROW + 1;
    if (numRows <= 0) {
        cudaMemset(output, 0, sizeof(int));
        return;
    }

    int* row_sums;
    cudaMalloc(&row_sums, numRows * sizeof(int));

    dim3 block1(256);
    dim3 grid1(numRows);
    size_t shared_mem_size = 256 * sizeof(int);
    row_sum_kernel<<<grid1, block1, shared_mem_size>>>(input, row_sums, N, M, S_ROW, E_ROW, S_COL, E_COL);

    dim3 block2(1024);
    dim3 grid2(1);
    total_sum_kernel<<<grid2, block2>>>(row_sums, output, numRows);

    cudaDeviceSynchronize();
    cudaFree(row_sums);
}