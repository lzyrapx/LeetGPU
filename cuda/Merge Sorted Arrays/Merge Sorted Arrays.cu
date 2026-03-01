#include <cuda_runtime.h>

// A, B 已经有序
__global__ void merge_kernel(const float* A, const float* B, float* C, int M, int N) {
    int total = M + N;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    // 对于位置 k，它肯定是原数组 A 的前 i 个元素和原数组 B 的前 j 个元素组成，且满足 i + j = k
    int k = idx;
    if (k >= total) return;
    
    // [max(0, k - N), min(k, M)]
    int l = (k - N <= 0) ? 0 : (k - N);
    int r = (k < M) ? k : M;

    // 二分 i
    while (l < r) {
        int mid = l + (r - l) / 2;
        int j = k - mid;
        if (A[mid] < B[j - 1]) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }

    // i + j = k
    int i = l, j = k - l;

    float val = 0.0f;
    if (i == M) {
        val = B[j];
    } else if (j == N) {
        val = A[i];
    } else {
        val = min(A[i], B[j]);
    }
    C[k] = val;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    const int BLOCK_SIZE = 16;
    int total = M + N;
    if (total == 0) return;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);  // 只需要 x 维度，y 维度设为 1
    merge_kernel<<<grid, block>>>(A, B, C, M, N);
}
