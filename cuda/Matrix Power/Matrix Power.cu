// https://leetgpu.com/challenges/matrix-power

#include "solve.h"
#include <cuda_runtime.h>

__global__ void identityKernel(float *mat, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        mat[row * N + col] = (row == col) ? 1.0f : 0.0f;
    }
}

__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void solve(const float* input, float* output, int N, int P) {
    if (P == 0) {
        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (N + 15) / 16);
        identityKernel<<<blocks, threads>>>(output, N);
        cudaDeviceSynchronize();
        return;
    }

    if (P == 1) {
        cudaMemcpy(output, input, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
        return;
    }

    float *d_buf1, *d_buf2, *d_buf3;
    cudaMalloc(&d_buf1, N * N * sizeof(float));
    cudaMalloc(&d_buf2, N * N * sizeof(float));
    cudaMalloc(&d_buf3, N * N * sizeof(float));

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    identityKernel<<<blocks, threads>>>(d_buf1, N);
    cudaMemcpy(d_buf2, input, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    float *d_result = d_buf1;
    float *d_base = d_buf2;
    float *d_temp = d_buf3;

    int exponent = P;
    while (exponent) {
        if (exponent & 1) {
            matrixMultiplyKernel<<<blocks, threads>>>(d_result, d_base, d_temp, N);
            cudaDeviceSynchronize();
            float *tmp = d_result;
            d_result = d_temp;
            d_temp = tmp;
        }
        exponent >>= 1;
        if (exponent) {
            matrixMultiplyKernel<<<blocks, threads>>>(d_base, d_base, d_temp, N);
            cudaDeviceSynchronize();
            float *tmp = d_base;
            d_base = d_temp;
            d_temp = tmp;
        }
    }

    cudaMemcpy(output, d_result, N * N * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFree(d_buf3);
}