#include <cuda_runtime.h>

__global__ void sliding_window_kernel(const int* input, int* output, int N, int window_size) {
    int current_sum = 0;
    for (int i = 0; i < window_size; i++) {
        current_sum += input[i];
    }
    int max_sum = current_sum;
    for (int i = window_size; i < N; i++) {
        current_sum = current_sum - input[i - window_size] + input[i];
        if (current_sum > max_sum) {
            max_sum = current_sum;
        }
    }
    *output = max_sum;
}

extern "C" void solve(const int* input, int* output, int N, int window_size) {
    sliding_window_kernel<<<1, 1>>>(input, output, N, window_size);
    cudaDeviceSynchronize();
}