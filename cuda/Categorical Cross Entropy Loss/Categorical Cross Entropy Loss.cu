#include <cuda_runtime.h>

#define WARP_SIZE 32

__global__ void loss_kernel(const float* logits, float* loss, const int* true_labels, int N, int C) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < N) {
        float sum = 0.0f;
        int base = index * C;
        for (int i = 0; i < C; i++) {
            sum += __expf(logits[base + i]);
        }
        sum = __logf(sum);
        sum -= logits[base + true_labels[index]];
        loss[index] = sum;
    }
} 

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    val = warpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void reduce_sum_kernel(const float* input, float* global_sum, int N) {
    float sum_val = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        sum_val += input[i];
    }
    sum_val = blockReduceSum(sum_val);
    if (threadIdx.x == 0) {
        atomicAdd(global_sum, sum_val / N);
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    float* d_loss;
    cudaMalloc(&d_loss, N * sizeof(float));
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 1024) {
        blocks_per_grid = 1024;
    }
    loss_kernel<<<blocks_per_grid, threads_per_block>>>(logits, d_loss, true_labels, N, C);
    reduce_sum_kernel<<<blocks_per_grid, threads_per_block>>>(d_loss, loss, N);
    cudaDeviceSynchronize();
    cudaFree(d_loss);
}
