#include <cuda_runtime.h>

__global__ void squared_diff_and_reduce(const float* predictions, const float* targets, float* partial_sums, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = 0.0f;
    if (i < N) {
        float diff = predictions[i] - targets[i];
        sdata[tid] = diff * diff;
    }
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_sum(float* input, float* output, int numElements) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < numElements) ? input[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void compute_mse(float* sum, float* mse, int N) {
    *mse = *sum / N;
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    const int BLOCK_SIZE = 256;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* d_partial_sums;
    cudaMalloc(&d_partial_sums, gridSize * sizeof(float));

    squared_diff_and_reduce<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(predictions, targets, d_partial_sums, N);

    int curSize = gridSize;
    float* curInput = d_partial_sums;
    while (curSize > 1) {
        gridSize = (curSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float* d_output;
        cudaMalloc(&d_output, gridSize * sizeof(float));
        reduce_sum<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(curInput, d_output, curSize);
        cudaFree(curInput);
        curInput = d_output;
        curSize = gridSize;
    }

    compute_mse<<<1, 1>>>(curInput, mse, N);
    cudaFree(curInput);
}