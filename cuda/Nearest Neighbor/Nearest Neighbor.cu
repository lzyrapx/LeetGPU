#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void nearest_neighbor_kernel(const float* points, int* indices, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) 
        return;
    float x1 = points[3 * idx];
    float y1 = points[3 * idx + 1];
    float z1 = points[3 * idx + 2];
    float min_dis_square = FLT_MAX;  // init to max to find mininum
    int nearest_idx = -1;  // init to invalid idx
    for (int i = 0; i < N; i++) {
        if (idx == i) continue;
        float x2 = points[3 * i];
        float y2 = points[3 * i + 1];
        float z2 = points[3 * i + 2];
        
        float dx = x1 - x2;
        float dy = y1 - y2;
        float dz = z1 - z2;
        
        float dis_square = dx * dx + dy * dy + dz * dz;
        
        if (dis_square < min_dis_square) {
            min_dis_square = dis_square;
            nearest_idx = i;
        }
    }
    // write to global memory
    indices[idx] = nearest_idx;
}

extern "C" void solve(const float* points, int* indices, int N) {
    if (N <= 0) 
        return;
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    nearest_neighbor_kernel<<<grid, block>>>(points, indices, N);
    cudaDeviceSynchronize();
}