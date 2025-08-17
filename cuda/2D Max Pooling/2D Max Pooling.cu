#include <cuda_runtime.h>

__global__ void max_pooling_2d(const float* input, float* output,
                                int N, int C, int H, int W,
                               int kernel_size, int stride, int padding) {
    // output dimensions
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    
    // combined index for batch and channel
    int n = blockIdx.z / C;
    int c = blockIdx.z % C;

    // output width and height indices
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (w_out >= W_out || h_out >= H_out)  return;

    // starting positions in the original input (considering padding)
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // initialize max value to negative infinity
    float max_val = __int_as_float(0xff800000); // -inf

    // iterate over the pooling window
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int h_orig = h_start + i;
            int w_orig = w_start + j;
            // check if the current position is within the original input bounds
            if (h_orig >= 0 && h_orig < H && w_orig >= 0 && w_orig < W) {
                int input_idx = n * C * H * W + c * H * W + h_orig * W + w_orig;
                float val = input[input_idx];
                if (val > max_val) 
                    max_val = val;
            }
        }
    }

    // compute the output index and store the result
    int output_idx = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out;
    output[output_idx] = max_val;

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output,
                      int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    // output dimensions
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    if (H_out <= 0 || W_out <= 0) return;

    // block dimensions (16x16 threads)
    dim3 blockDim(16, 16);
    int gridX = (W_out + blockDim.x - 1) / blockDim.x;
    int gridY = (H_out + blockDim.y - 1) / blockDim.y;
    int gridZ = N * C;
    
    dim3 gridDim(gridX, gridY, gridZ);
    max_pooling_2d<<<gridDim, blockDim>>>(input, output, N, C, H, W, kernel_size, stride, padding);
    cudaDeviceSynchronize();
}