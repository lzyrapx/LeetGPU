https://leetgpu.com/challenges/3d-convolution

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* kernel, float* output,
                              int input_depth, int input_rows, int input_cols,
                              int kernel_depth, int kernel_rows, int kernel_cols,
                              int output_depth, int output_rows, int output_cols) {
    // 获取当前线程对应的输出坐标(i, j, k)
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= output_depth || j >= output_rows || k >= output_cols) {
        return;
    }

    float sum = 0.0f;

    // 遍历kernel的三个维度
    for (int d = 0; d < kernel_depth; ++d) {
        for (int r = 0; r < kernel_rows; ++r) {
            for (int c = 0; c < kernel_cols; ++c) {
                // 计算输入元素的坐标
                int input_d = i + d;
                int input_r = j + r;
                int input_c = k + c;

                // 计算输入和kernel的线性索引
                int input_idx = input_d * (input_rows * input_cols) + input_r * input_cols + input_c;
                int kernel_idx = d * (kernel_rows * kernel_cols) + r * kernel_cols + c;

                sum += input[input_idx] * kernel[kernel_idx];
            }
        }
    }

    // 计算输出的线性索引并存储结果
    int output_idx = i * (output_rows * output_cols) + j * output_cols + k;
    output[output_idx] = sum;
}

void solve(const float* input, const float* kernel, float* output,
           int input_depth, int input_rows, int input_cols,
           int kernel_depth, int kernel_rows, int kernel_cols) {
    // 计算输出尺寸
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    if (output_depth <= 0 || output_rows <= 0 || output_cols <= 0) {
        return; // 无效输出尺寸
    }

    // 分配设备内存
    float *d_input, *d_kernel, *d_output;
    size_t input_size = input_depth * input_rows * input_cols;
    size_t kernel_size = kernel_depth * kernel_rows * kernel_cols;
    size_t output_size = output_depth * output_rows * output_cols;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // 配置线程块和网格
    dim3 block(8, 8, 4); // 线程块维度
    dim3 grid(
        (output_cols + block.x - 1) / block.x,
        (output_rows + block.y - 1) / block.y,
        (output_depth + block.z - 1) / block.z
    );

    // 启动核函数
    conv3d_kernel<<<grid, block>>>(d_input, d_kernel, d_output,
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        output_depth, output_rows, output_cols);

    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

    // 拷贝结果回主机
    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
```