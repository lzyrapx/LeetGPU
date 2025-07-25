// https://leetgpu.com/challenges/logistic-regression

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
// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
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
        logistic_regression_kernel<<<gridDim, blockDim>>>(X, y, beta, d_grad, n, d);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // 更新权重 (直接在 device 上更新，避免 host-device 拷贝)
        update_weights_kernel<<<updateGrid, blockDim>>>(beta, d_grad, learning_rate, n, d);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d_grad));
}

/*
Test failed! Here are the inputs:
X = [[0.125, 0.6579999923706055, 0.6230000257492065], [-0.8019999861717224, -0.23399999737739563, -0.8579999804496765], [0.9290000200271606, 0.04399999976158142, 0.4740000069141388]]
y = [1.0, 0.0, 1.0]
n_samples = 3
n_features = 3
Mismatch in 'beta'
Expected: [7.599221229553223, 6.970425128936768, 9.579840660095215]
Got: [4.2254252433776855, 3.331865072250366, 5.195430278778076]
*/
