#include <cuda_runtime.h>

const int threads = 128;
const int warp_size = 32;

// how many elements each thread processes; increases work per thread
const int elements_per_thread = 4;

__global__ void mean_squared_error(const float* predictions, const float* targets, float* mse, int N) {
    __shared__ float sdata[threads / warp_size];
    
    int idx = blockIdx.x * blockDim.x * elements_per_thread + threadIdx.x;

    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    float val = 0.0f;
    for (int i = 0; i < elements_per_thread; i++) {
        int local_idx = idx + i * blockDim.x;
        if (local_idx < N) {
            float delta = predictions[local_idx] - targets[local_idx];
            val += delta * delta;
        }
    }

    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane_id == 0) sdata[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (threadIdx.x < threads / warp_size) ? sdata[threadIdx.x] : 0.0f;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) atomicAdd(mse, val / N);
    }
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    int blocks = (N + threads * elements_per_thread - 1) / (threads * elements_per_thread);
    mean_squared_error<<<blocks, threads>>>(predictions, targets, mse, N);
}
