// https://leetgpu.com/challenges/ordinary-least-squares-regression

#include "solve.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void xtx_kernel(const float* X, float* C, int n_samples, int n_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n_features && j < n_features) {
        float sum = 0.0f;
        for (int k = 0; k < n_samples; k++) {
            float xi = X[k * n_features + i];
            float xj = X[k * n_features + j];
            sum += xi * xj;
        }
        C[i * n_features + j] = sum;
    }
}

__global__ void xty_kernel(const float* X, const float* y, float* b, int n_samples, int n_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_features) {
        float sum = 0.0f;
        for (int k = 0; k < n_samples; k++) {
            sum += X[k * n_features + i] * y[k];
        }
        b[i] = sum;
    }
}

__global__ void cholesky_kernel(float* A, int n) {
    int tid = threadIdx.x;
    for (int i = 0; i < n; i++) {
        if (tid == i) {
            float sum = 0.0f;
            for (int k = 0; k < i; k++) {
                float lik = A[i * n + k];
                sum += lik * lik;
            }
            A[i * n + i] = sqrtf(A[i * n + i] - sum);
        }
        __syncthreads();

        if (tid > i && tid < n) {
            float sum = 0.0f;
            for (int k = 0; k < i; k++) {
                sum += A[tid * n + k] * A[i * n + k];
            }
            A[tid * n + i] = (A[tid * n + i] - sum) / A[i * n + i];
        }
        __syncthreads();
    }
}

__global__ void forward_substitution_kernel(const float* L, float* b, int n) {
    extern __shared__ float s_w[];
    int tid = threadIdx.x;

    if (tid < n) {
        s_w[tid] = b[tid];
    }
    __syncthreads();

    for (int i = 0; i < n; i++) {
        if (tid == i) {
            float sum = 0.0f;
            for (int j = 0; j < i; j++) {
                sum += L[i * n + j] * s_w[j];
            }
            s_w[i] = (s_w[i] - sum) / L[i * n + i];
        }
        __syncthreads();
    }

    if (tid < n) {
        b[tid] = s_w[tid];
    }
}

__global__ void backward_substitution_kernel(const float* L, const float* w, float* beta, int n) {
    extern __shared__ float s_beta[];
    int tid = threadIdx.x;

    if (tid < n) {
        s_beta[tid] = 0.0f;
    }
    __syncthreads();

    for (int i = n - 1; i >= 0; i--) {
        if (tid == i) {
            float sum = 0.0f;
            for (int j = i + 1; j < n; j++) {
                sum += L[j * n + i] * s_beta[j];
            }
            s_beta[i] = (w[i] - sum) / L[i * n + i];
        }
        __syncthreads();
    }

    if (tid < n) {
        beta[tid] = s_beta[tid];
    }
}

// X, y, beta are device pointers
void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    if (n_features == 0) {
        return;
    }

    float *d_A, *d_b;
    cudaMalloc(&d_A, n_features * n_features * sizeof(float));
    cudaMalloc(&d_b, n_features * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((n_features + 15) / 16, (n_features + 15) / 16);
    xtx_kernel<<<grid, block>>>(X, d_A, n_samples, n_features);

    dim3 block1(256);
    dim3 grid1((n_features + 255) / 256);
    xty_kernel<<<grid1, block1>>>(X, y, d_b, n_samples, n_features);
    cudaDeviceSynchronize();

    cholesky_kernel<<<1, n_features>>>(d_A, n_features);
    cudaDeviceSynchronize();

    size_t shared_mem_size = n_features * sizeof(float);
    forward_substitution_kernel<<<1, n_features, shared_mem_size>>>(d_A, d_b, n_features);
    cudaDeviceSynchronize();

    backward_substitution_kernel<<<1, n_features, shared_mem_size>>>(d_A, d_b, beta, n_features);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_b);
}
