// https://leetgpu.com/challenges/matrix-multiplication

#include "solve.h"
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    // Calculate the row and column index of the element in the output matrix C
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the current thread is within the bounds of the output matrix
    if (row < M && col < K) {
        float value = 0.0f;
        // C[i][j] = \sum_{k=0}^{N-1} A[i][k] * B[k][j]
        // A 是一个 M x N 的矩阵，B 是一个 N x K 的矩阵，C 是一个 M x K 的矩阵
        for (int k = 0; k < N; k++) {
            // Iterate over the shared dimension N to compute the dot product
            value += A[row * N + k] * B[k * K + col];
        }
        // Store the result in the output matrix C
        C[row * K + col] = value;
    }
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy input data from host to device
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the kernel
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}