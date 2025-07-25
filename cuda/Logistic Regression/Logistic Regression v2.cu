#include <cuda_runtime.h>
#include <math.h>

__global__ void matvec_kernel(const float* X, const float* beta, float* z, int n_samples, int n_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_samples) {
        float sum = 0.0f;
        for (int j = 0; j < n_features; j++) {
            sum += X[i * n_features + j] * beta[j];
        }
        z[i] = sum;
    }
}

__global__ void sigmoid_residual_kernel(const float* z, const float* y, float* r, int n_samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_samples) {
        float p = 1.0f / (1.0f + expf(-z[i]));
        r[i] = p - y[i];
    }
}

__global__ void grad_kernel(const float* X, const float* r, float* grad, int n_samples, int n_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n_features) {
        float sum = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            sum += X[i * n_features + j] * r[i];
        }
        grad[j] = sum;
    }
}

__global__ void update_kernel(float* beta, const float* grad, float alpha, int n_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n_features) {
        beta[j] = beta[j] - alpha * grad[j];
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    cudaMemset(beta, 0, n_features * sizeof(float));

    float *d_z, *d_r, *d_grad;
    cudaMalloc(&d_z, n_samples * sizeof(float));
    cudaMalloc(&d_r, n_samples * sizeof(float));
    cudaMalloc(&d_grad, n_features * sizeof(float));

    const float alpha = 0.01f;
    const int max_iter = 50000;

    const int block_size = 256;
    dim3 grid_samples((n_samples + block_size - 1) / block_size);
    dim3 grid_features((n_features + block_size - 1) / block_size);

    for (int iter = 0; iter < max_iter; iter++) {
        matvec_kernel<<<grid_samples, block_size>>>(X, beta, d_z, n_samples, n_features);
        cudaDeviceSynchronize();

        sigmoid_residual_kernel<<<grid_samples, block_size>>>(d_z, y, d_r, n_samples);
        cudaDeviceSynchronize();

        grad_kernel<<<grid_features, block_size>>>(X, d_r, d_grad, n_samples, n_features);
        cudaDeviceSynchronize();

        update_kernel<<<grid_features, block_size>>>(beta, d_grad, alpha, n_features);
        cudaDeviceSynchronize();
    }

    cudaFree(d_z);
    cudaFree(d_r);
    cudaFree(d_grad);
}


/*
Test failed! Here are the inputs:
X = [[0.125, 0.6579999923706055, 0.6230000257492065], [-0.8019999861717224, -0.23399999737739563, -0.8579999804496765], [0.9290000200271606, 0.04399999976158142, 0.4740000069141388]]
y = [1.0, 0.0, 1.0]
n_samples = 3
n_features = 3
Mismatch in 'beta'
Expected: [7.599221229553223, 6.970425128936768, 9.579840660095215]
Got: [4.470566272735596, 3.5762758255004883, 5.526671886444092]
*/