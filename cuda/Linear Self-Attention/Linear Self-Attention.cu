#include <cuda_runtime.h>
#include <math.h>

__global__ void kernel_phi(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = (x > 0) ? (x + 1.0f) : expf(x);
    }
}

__global__ void kernel_reduce_z(const float* phi_K, float* z, int M, int d) {
    int col = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < M; i += blockDim.x) {
        sum += phi_K[i * d + col];
    }
    
    __shared__ float sh_sum[256];
    sh_sum[tid] = sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sh_sum[tid] += sh_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        z[col] = sh_sum[0];
    }
}

__global__ void kernel_compute_S(const float* phi_K, const float* V, float* S, int M, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < d && j < d) {
        float temp = 0.0f;
        for (int k = 0; k < M; k++) {
            temp += phi_K[k * d + i] * V[k * d + j];
        }
        S[i * d + j] = temp;
    }
}

__global__ void kernel_compute_numerator(const float* phi_Q, const float* S, float* numerator, int M, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < d) {
        float temp = 0.0f;
        for (int k = 0; k < d; k++) {
            temp += phi_Q[row * d + k] * S[k * d + col];
        }
        numerator[row * d + col] = temp;
    }
}

__global__ void kernel_compute_denominator(const float* phi_Q, const float* z, float* denominator, int M, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float temp = 0.0f;
        for (int k = 0; k < d; k++) {
            temp += phi_Q[row * d + k] * z[k];
        }
        denominator[row] = temp;
    }
}

__global__ void kernel_divide(const float* numerator, const float* denominator, float* output, int M, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * d) {
        int row = idx / d;
        output[idx] = numerator[idx] / denominator[row];
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    float *phi_K, *phi_Q, *z, *S, *numerator, *denominator;
    cudaMalloc(&phi_K, M * d * sizeof(float));
    cudaMalloc(&phi_Q, M * d * sizeof(float));
    cudaMalloc(&z, d * sizeof(float));
    cudaMalloc(&S, d * d * sizeof(float));
    cudaMalloc(&numerator, M * d * sizeof(float));
    cudaMalloc(&denominator, M * sizeof(float));

    dim3 block(256);
    dim3 grid_phi((M * d + block.x - 1) / block.x);
    kernel_phi<<<grid_phi, block>>>(K, phi_K, M * d);
    kernel_phi<<<grid_phi, block>>>(Q, phi_Q, M * d);

    kernel_reduce_z<<<d, 256>>>(phi_K, z, M, d);

    dim3 block_S(16, 16);
    dim3 grid_S((d + 15) / 16, (d + 15) / 16);
    kernel_compute_S<<<grid_S, block_S>>>(phi_K, V, S, M, d);

    dim3 block_num(16, 16);
    dim3 grid_num((M + 15) / 16, (d + 15) / 16);
    kernel_compute_numerator<<<grid_num, block_num>>>(phi_Q, S, numerator, M, d);

    kernel_compute_denominator<<<(M + 255) / 256, 256>>>(phi_Q, z, denominator, M, d);

    kernel_divide<<<grid_phi, block>>>(numerator, denominator, output, M, d);

    cudaFree(phi_K);
    cudaFree(phi_Q);
    cudaFree(z);
    cudaFree(S);
    cudaFree(numerator);
    cudaFree(denominator);
}