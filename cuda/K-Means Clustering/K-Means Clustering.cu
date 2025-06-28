// https://leetgpu.com/challenges/k-means-clustering

#include "solve.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void assign_clusters(const float* data_x, const float* data_y, int* labels, 
                                const float* centroids_x, const float* centroids_y, 
                                int k, int sample_size) {
    extern __shared__ float s_centroids[];
    float* s_x = s_centroids;
    float* s_y = s_centroids + k;

    int tid = threadIdx.x;
    for (int i = tid; i < k; i += blockDim.x) {
        if (i < k) {
            s_x[i] = centroids_x[i];
            s_y[i] = centroids_y[i];
        }
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sample_size) return;

    float x = data_x[idx];
    float y = data_y[idx];
    float min_dist_sq = FLT_MAX;
    int min_idx = -1;

    for (int c = 0; c < k; ++c) {
        float dx = x - s_x[c];
        float dy = y - s_y[c];
        float dist_sq = dx * dx + dy * dy;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            min_idx = c;
        }
    }
    labels[idx] = min_idx;
}

__global__ void compute_sums(const float* data_x, const float* data_y, const int* labels,
                             float* sum_x, float* sum_y, int* count, int sample_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sample_size) return;

    int c = labels[idx];
    atomicAdd(&sum_x[c], data_x[idx]);
    atomicAdd(&sum_y[c], data_y[idx]);
    atomicAdd(&count[c], 1);
}

__global__ void update_centroids(float* new_centroid_x, float* new_centroid_y,
                                 const float* sum_x, const float* sum_y, const int* count,
                                 const float* old_centroid_x, const float* old_centroid_y, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;

    if (count[c] > 0) {
        new_centroid_x[c] = sum_x[c] / count[c];
        new_centroid_y[c] = sum_y[c] / count[c];
    } else {
        new_centroid_x[c] = old_centroid_x[c];
        new_centroid_y[c] = old_centroid_y[c];
    }
}

__global__ void compute_distances(const float* old_x, const float* old_y,
                                  const float* new_x, const float* new_y,
                                  float* distances, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;

    float dx = new_x[c] - old_x[c];
    float dy = new_y[c] - old_y[c];
    distances[c] = sqrtf(dx * dx + dy * dy);
}

void solve(const float* data_x, const float* data_y, int* labels,
           float* initial_centroid_x, float* initial_centroid_y,
           float* final_centroid_x, float* final_centroid_y,
           int sample_size, int k, int max_iterations) {
    // Copy initial centroids to final centroids (output)
    cudaMemcpy(final_centroid_x, initial_centroid_x, k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(final_centroid_y, initial_centroid_y, k * sizeof(float), cudaMemcpyDeviceToDevice);

    // Allocate device memory
    float *d_old_centroid_x, *d_old_centroid_y;
    cudaMalloc(&d_old_centroid_x, k * sizeof(float));
    cudaMalloc(&d_old_centroid_y, k * sizeof(float));

    float *d_sum_x, *d_sum_y;
    int *d_count;
    cudaMalloc(&d_sum_x, k * sizeof(float));
    cudaMalloc(&d_sum_y, k * sizeof(float));
    cudaMalloc(&d_count, k * sizeof(int));

    float *d_distances;
    cudaMalloc(&d_distances, k * sizeof(float));

    float *h_distances = (float*)malloc(k * sizeof(float));

    // Main loop
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Copy current centroids to old
        cudaMemcpy(d_old_centroid_x, final_centroid_x, k * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_old_centroid_y, final_centroid_y, k * sizeof(float), cudaMemcpyDeviceToDevice);

        // Assign clusters
        int blockSize = 256;
        int gridSize = (sample_size + blockSize - 1) / blockSize;
        int sharedMemSize = 2 * k * sizeof(float);
        assign_clusters<<<gridSize, blockSize, sharedMemSize>>>(data_x, data_y, labels, d_old_centroid_x, d_old_centroid_y, k, sample_size);
        cudaDeviceSynchronize();

        // Reset sums and counts
        cudaMemset(d_sum_x, 0, k * sizeof(float));
        cudaMemset(d_sum_y, 0, k * sizeof(float));
        cudaMemset(d_count, 0, k * sizeof(int));

        // Compute sums
        compute_sums<<<gridSize, blockSize>>>(data_x, data_y, labels, d_sum_x, d_sum_y, d_count, sample_size);
        cudaDeviceSynchronize();

        // Update centroids
        int updateBlockSize = 256;
        int updateGridSize = (k + updateBlockSize - 1) / updateBlockSize;
        update_centroids<<<updateGridSize, updateBlockSize>>>(final_centroid_x, final_centroid_y,
                                                               d_sum_x, d_sum_y, d_count,
                                                               d_old_centroid_x, d_old_centroid_y, k);
        cudaDeviceSynchronize();

        // Compute distances between old and new centroids
        compute_distances<<<updateGridSize, updateBlockSize>>>(d_old_centroid_x, d_old_centroid_y,
                                                                final_centroid_x, final_centroid_y,
                                                                d_distances, k);
        cudaDeviceSynchronize();

        // Check convergence
        cudaMemcpy(h_distances, d_distances, k * sizeof(float), cudaMemcpyDeviceToHost);
        float max_move = 0.0f;
        for (int i = 0; i < k; ++i) {
            if (h_distances[i] > max_move) {
                max_move = h_distances[i];
            }
        }
        if (max_move < 0.0001f) {
            break;
        }
    }

    // Cleanup
    free(h_distances);
    cudaFree(d_old_centroid_x);
    cudaFree(d_old_centroid_y);
    cudaFree(d_sum_x);
    cudaFree(d_sum_y);
    cudaFree(d_count);
    cudaFree(d_distances);
}

// solution 2

#include "solve.h"
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
    
    for (int c = 0; c < k; ++c) {
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
void solve(const float* data_x, const float* data_y, int* labels,
           float* initial_centroid_x, float* initial_centroid_y,
           float* final_centroid_x, float* final_centroid_y,
           int sample_size, int k, int max_iterations) {

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
    
    for (int iter = 0; iter < max_iterations; ++iter) {
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
        for (int c = 0; c < k; ++c) {
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
