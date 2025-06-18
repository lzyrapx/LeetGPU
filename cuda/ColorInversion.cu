
// https://leetgpu.com/challenges/color-inversion

#include "solve.h"
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    // 图像数据以一维数组存储，每个像素包含连续的4个字节（RGBA）
    // 定位像素内存为位置
    unsigned char* pixel = image + (idx * 4);
    pixel[0] = 255 - pixel[0]; // Invert Red
    pixel[1] = 255 - pixel[1]; // Invert Green
    pixel[2] = 255 - pixel[2]; // Invert Blue
    // Alpha分量（pixel[3]）保持不变
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;  // 每个块256个线程
    // 计算需要的块数（向上取整）
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}