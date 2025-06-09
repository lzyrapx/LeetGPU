#include "solve.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>

#define BLOCK_SIZE 256
#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)

void check_cuda_error(cudaError_t err, const char* func, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " - " << func 
                  << " failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void logistic_regression_kernel(const float *X, const float *y, 
                                          float *weights, float *grad, 
                                          int n, int d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float prediction = 0.0f;
        for (int j = 0; j < d; j++) {
            prediction += X[tid * d + j] * weights[j];
        }
        prediction = sigmoid(prediction);
        
        float error = prediction - y[tid];
        
        for (int j = 0; j < d; j++) {
            atomicAdd(&grad[j], error * X[tid * d + j]);
        }
    }
}

// 更新权重
__global__ void update_weights_kernel(float *weights, const float *grad, 
                                    float lr, float n, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d) {
        weights[idx] -= lr * grad[idx] / n;
    }
}

// X, y, beta are device pointers
// n_samples: 样本数量, n_features: 特征数量
void solve(const float* d_X, const float* d_y, float* d_beta, 
          int n_samples, int n_features) {
    int n = n_samples;
    int d = n_features;

    const float learning_rate = 0.01f;
    const int epochs = 100000;

    float *d_grad;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad, d * sizeof(float)));

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 updateGrid((d + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 梯度下降循环
    for (int iter = 0; iter < epochs; iter++) {
        // 重置梯度
        CHECK_CUDA_ERROR(cudaMemset(d_grad, 0, d * sizeof(float)));
        
        // 计算预测值和梯度
        logistic_regression_kernel<<<gridDim, blockDim>>>(d_X, d_y, d_beta, d_grad, n, d);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // 更新权重 (直接在 device 上更新，避免 host-device 拷贝)
        update_weights_kernel<<<updateGrid, blockDim>>>(d_beta, d_grad, learning_rate, n, d);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d_grad));
}
