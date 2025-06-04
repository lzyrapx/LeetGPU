// https://leetgpu.com/challenges/logistic-regression

#include "solve.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// sigmoid function
__global__ void sigmoid_kernel(float *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = 1.0f / (1.0f + expf(-z[i]));
    }
}

void logistic_regression_kernel(float *d_X, float *d_y, float *d_w, 
                                int n, int d, float learning_rate, int epochs) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;
    float neg_learning_rate = -learning_rate;

    float *d_pred, *d_grad;
    cudaMalloc(&d_pred, n * sizeof(float));  // Stores predictions
    cudaMalloc(&d_grad, d * sizeof(float));  // Stores gradient

    for (int epoch = 0; epoch < epochs; epoch++) {
        cublasSgemv(handle, CUBLAS_OP_T, n, d, &alpha, d_X, d, d_w, 1, &beta, d_pred, 1);

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        sigmoid_kernel<<<numBlocks, blockSize>>>(d_pred, n);

        float minus_one = -1.0f;
        cublasSaxpy(handle, n, &minus_one, d_y, 1, d_pred, 1);

        cublasSgemv(handle, CUBLAS_OP_N, d, n, &alpha, d_X, n, d_pred, 1, &beta, d_grad, 1);

        cublasSaxpy(handle, d, &neg_learning_rate, d_grad, 1, d_w, 1);

        if (epoch % 100 == 0) {
            float loss = 0.0f;
            cublasSnrm2(handle, n, d_pred, 1, &loss);
            cout << "Epoch: " << epoch << " Loss: " << (loss / n) << endl;
        }
    }

    cudaFree(d_pred);
    cudaFree(d_grad);
    cublasDestroy(handle);
}

// X, y, beta are device pointers
// n_samples: 样本数量, n_features: 特征数量
void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    int d = n_features;
    int n = n_samples;
    const float learning_rate = 0.1f;
    const int epochs = 1000;

    // X shape: [n, d]
    // y shape: [n, 1]
    // beta shape: [d, 1]

    cudaMalloc(&beta, d * sizeof(float));
    // init weights to zero
    cudaMemset(beta, 0, d * sizeof(float));
    logistic_regression_kernel(X, y, beta, n, d, learning_rate, epochs);
    cudaDeviceSynchronize();
}
