#include <cuda_runtime.h>
#include <cmath>

__device__ float dot_product(const float* a, const float* b, int d) {
    float result = 0.0f;
    for (int i = 0; i < d; i++) {
        result += a[i] * b[i];
    }
    return result;
}

__global__ void sliding_window_attention_kernel(
    const float* Q, const float* K, const float* V, float* output, 
    int M, int d, int window_size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    
    // each thread handles one query position
    const float* query_i = Q + i * d;
    
    // window boundaries
    int start = max(0, i - window_size);
    int end = min(M - 1, i + window_size);
    int window_length = end - start + 1;
    
    float max_score = -1e20f;
    float scores[65];  // maximum window size is 2*32+1 = 65
    
    for (int j = start, idx = 0; j <= end; j++, idx++) {
        const float* key_j = K + j * d;
        float score = dot_product(query_i, key_j, d) / sqrtf(static_cast<float>(d));
        scores[idx] = score;
        if (score > max_score) {
            max_score = score;
        }
    }
    
    float exp_sum = 0.0f;
    float exps[65];  // maximum window size is 2*32+1 = 65

    for (int idx = 0; idx < window_length; idx++) {
        float exp_val = expf(scores[idx] - max_score);
        exps[idx] = exp_val;
        exp_sum += exp_val;
    }
    
    for (int k = 0; k < d; k++) {
        output[i * d + k] = 0.0f;
    }
    
    for (int j = start, idx = 0; j <= end; j++, idx++) {
        float weight = exps[idx] / exp_sum;
        const float* value_j = V + j * d;
        
        for (int k = 0; k < d; k++) {
            output[i * d + k] += weight * value_j[k];
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size) {
    int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;
    sliding_window_attention_kernel<<<grid_size, block_size>>>(Q, K, V, output, M, d, window_size);
    cudaDeviceSynchronize();
}
