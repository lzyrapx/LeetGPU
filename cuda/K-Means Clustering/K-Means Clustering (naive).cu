#include <cuda_runtime.h>

__global__ void assign_clusters(const float* data_x, const float* data_y, int* labels,
                                const float* centroids_x, const float* centroids_y,
                                float* sum_x, float* sum_y, int* counts,
                                int sample_size, int k) {
                                    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sample_size) return;
    
    float x = data_x[idx];
    float y = data_y[idx];
    float min_dist = INFINITY;
    int nearest = 0;
    
    for (int c = 0; c < k; c++) {
        float dx = x - centroids_x[c];
        float dy = y - centroids_y[c];
        float dist = dx*dx + dy*dy;
        if (dist < min_dist) {
            min_dist = dist;
            nearest = c;
        }
    }
    
    labels[idx] = nearest;
    atomicAdd(&sum_x[nearest], x);
    atomicAdd(&sum_y[nearest], y);
    atomicAdd(&counts[nearest], 1);
}

__global__ void compute_centroids(const float* old_x, const float* old_y,
                                  float* new_x, float* new_y,
                                  const float* sum_x, const float* sum_y,
                                  const int* counts, int k) {
    int c = threadIdx.x;
    if (c >= k) return;
    
    int count = counts[c];
    if (count > 0) {
        new_x[c] = sum_x[c] / count;
        new_y[c] = sum_y[c] / count;
    } else {
        new_x[c] = old_x[c];
        new_y[c] = old_y[c];
    }
}

__global__ void compute_deltas(const float* old_x, const float* old_y,
                               const float* new_x, const float* new_y,
                               float* delta_sq, int k) {
    int c = threadIdx.x;
    if (c >= k) return;
    
    float dx = new_x[c] - old_x[c];
    float dy = new_y[c] - old_y[c];
    delta_sq[c] = dx*dx + dy*dy;
}


// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) {

    unsigned int block_size = 256;
    unsigned int grid_size = (sample_size + block_size - 1) / block_size;
    
    float *d_sum_x, *d_sum_y, *d_new_centroid_x, *d_new_centroid_y;
    int *d_counts;
    float *d_delta_sq;
    
    cudaMalloc((void**)&d_sum_x, k * sizeof(float));
    cudaMalloc((void**)&d_sum_y, k * sizeof(float));
    cudaMalloc((void**)&d_counts, k * sizeof(int));
    cudaMalloc((void**)&d_new_centroid_x, k * sizeof(float));
    cudaMalloc((void**)&d_new_centroid_y, k * sizeof(float));
    cudaMalloc((void**)&d_delta_sq, k * sizeof(float));
    
    cudaMemcpy(final_centroid_x, initial_centroid_x, k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(final_centroid_y, initial_centroid_y, k * sizeof(float), cudaMemcpyDeviceToDevice);
    
    float *h_delta_sq = new float[k];
    
    for (int iter = 0; iter < max_iterations; iter++) {
        cudaMemset(d_sum_x, 0, k * sizeof(float));
        cudaMemset(d_sum_y, 0, k * sizeof(float));
        cudaMemset(d_counts, 0, k * sizeof(int));
        
        assign_clusters<<<grid_size, block_size>>>(
            data_x, data_y, labels, final_centroid_x, final_centroid_y,
            d_sum_x, d_sum_y, d_counts, sample_size, k
        );
        
        compute_centroids<<<1, k>>>(
            final_centroid_x, final_centroid_y,
            d_new_centroid_x, d_new_centroid_y,
            d_sum_x, d_sum_y, d_counts, k
        );
        
        compute_deltas<<<1, k>>>(
            final_centroid_x, final_centroid_y,
            d_new_centroid_x, d_new_centroid_y,
            d_delta_sq, k
        );
        
        cudaMemcpy(h_delta_sq, d_delta_sq, k * sizeof(float), cudaMemcpyDeviceToHost);
        float max_delta_sq = 0.0f;
        for (int c = 0; c < k; c++) {
            if (h_delta_sq[c] > max_delta_sq) {
                max_delta_sq = h_delta_sq[c];
            }
        }
        if (max_delta_sq <= 1e-8f) break;
        
        cudaMemcpy(final_centroid_x, d_new_centroid_x, k * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(final_centroid_y, d_new_centroid_y, k * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(d_sum_x);
    cudaFree(d_sum_y);
    cudaFree(d_counts);
    cudaFree(d_new_centroid_x);
    cudaFree(d_new_centroid_y);
    cudaFree(d_delta_sq);
    delete[] h_delta_sq;
}
