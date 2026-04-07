#include <cuda_runtime.h>

__global__ void  linear_recurrence_kernel(const float *a, const float *x, float *h, int B, int L) {
    int batch = blockIdx.x;  // 每个 block 处理一个 batch
    if (batch >= B) return;
    a += batch * L;
    x += batch * L;
    h += batch * L;
    h[0] = x[0];
    for (int i = 1; i < L; i++) {
        h[i] = a[i] * h[i - 1] + x[i];
    }
}

// a, x, h are device pointers
extern "C" void solve(const float* a, const float* x, float* h, int B, int L) {
    // 每个 block 一个线程，grid 大小等于 batch 数
    // B = batch 数
    linear_recurrence_kernel<<<B, 1>>>(a, x, h, B, L);
    cudaDeviceSynchronize();
}
