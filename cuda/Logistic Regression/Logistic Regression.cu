#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

constexpr int BLOCK_SIZE = 32;

__global__ void mat_vec_mul_kernel(const float* X, const float* beta, float* z, 
                               int n_samples, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        float sum = 0.0f;
        for (int j = 0; j < n_features; j++) {
            sum += X[idx * n_features + j] * beta[j];
        }
        z[idx] = sum;
    }
}

__global__ void sigmoid_kernel(float* z, float* p, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        p[idx] = 1.0f / (1.0f + expf(-z[idx]));
    }
}

__global__ void subtract_kernel(const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void compute_grad_part_kernel(const float* X, const float* diff, float* grad_temp, 
                                     int n_samples, int n_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n_features) {
        float sum = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            sum += X[i * n_features + j] * diff[i];
        }
        grad_temp[j] = sum;
    }
}

__global__ void update_grad_kernel(float* grad, const float* grad_temp, const float* beta, 
                                float lambda_reg, float inv_n_samples, int n_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n_features) {
        grad[j] = grad_temp[j] * inv_n_samples + lambda_reg * beta[j];
    }
}

__global__ void compute_w_kernel(const float* p, float* w, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        w[idx] = p[idx] * (1.0f - p[idx]);
    }
}

__global__ void diag_mul_kernel(const float* w, const float* X, float* temp, 
                             int n_samples, int n_features) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_samples && j < n_features) {
        temp[i * n_features + j] = w[i] * X[i * n_features + j];
    }
}

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, 
                                   int A_rows, int A_cols, int B_cols, float alpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < A_cols && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_rows; k++) {
            sum += A[k * A_cols + row] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum * alpha;
    }
}

__global__ void add_regularization_kernel(float* matrix, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        matrix[idx * n + idx] += value;
    }
}

__global__ void cholesky_decomp_kernel(float* A, float* L, int n) {
    // Serial Cholesky for small matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            
            if (i == j) {
                L[i * n + j] = sqrtf(A[i * n + i] - sum);
            } else {
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
}

__global__ void solve_forward_kernel(float* L, float* b, float* y, int n) {
    // Solve L * y = b (forward substitution)
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < i; j++) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i * n + i];
    }
}

__global__ void solve_backward_kernel(float* L, float* y, float* x, int n) {
    // Solve L^T * x = y (backward substitution)
    for (int i = n - 1; i >= 0; i--) {
        float sum = 0.0f;
        for (int j = i + 1; j < n; j++) {
            sum += L[j * n + i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i * n + i];
    }
}

__global__ void update_beta_kernel(float* beta, const float* delta, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_features) {
        beta[idx] -= delta[idx];
    }
}

extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    const float lambda_reg = 1e-6f / n_samples;
    const float inv_n_samples = 1.0f / n_samples;
    const int max_iter = 30;
    const float epsilon = 1e-6f;

    // Allocate device memory
    float *d_z, *d_p, *d_w, *d_diff;
    float *d_grad, *d_grad_temp, *d_Hessian, *d_L, *d_y_temp, *d_delta, *d_temp;
    
    cudaMalloc(&d_z, n_samples * sizeof(float));
    cudaMalloc(&d_p, n_samples * sizeof(float));
    cudaMalloc(&d_w, n_samples * sizeof(float));
    cudaMalloc(&d_diff, n_samples * sizeof(float));
    cudaMalloc(&d_grad, n_features * sizeof(float));
    cudaMalloc(&d_grad_temp, n_features * sizeof(float));
    cudaMalloc(&d_Hessian, n_features * n_features * sizeof(float));
    cudaMalloc(&d_L, n_features * n_features * sizeof(float));
    cudaMalloc(&d_y_temp, n_features * sizeof(float));
    cudaMalloc(&d_delta, n_features * sizeof(float));
    cudaMalloc(&d_temp, n_samples * n_features * sizeof(float));

    // Set kernel dimensions
    dim3 block1d(BLOCK_SIZE);
    dim3 grid_samples((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_features((n_features + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    dim3 block2d(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_temp(
        (n_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    dim3 grid_hessian(
        (n_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (n_features + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    for (int iter = 0; iter < max_iter; iter++) {
        // z = X * beta
        mat_vec_mul_kernel<<<grid_samples, block1d>>>(X, beta, d_z, n_samples, n_features);
        cudaDeviceSynchronize();

        // p = sigmoid(z)
        sigmoid_kernel<<<grid_samples, block1d>>>(d_z, d_p, n_samples);
        cudaDeviceSynchronize();

        // diff = p - y
        subtract_kernel<<<grid_samples, block1d>>>(d_p, y, d_diff, n_samples);
        cudaDeviceSynchronize();

        // grad = X^T * diff / n_samples + lambda_reg * beta
        compute_grad_part_kernel<<<grid_features, block1d>>>(X, d_diff, d_grad_temp, n_samples, n_features);
        cudaDeviceSynchronize();
        
        update_grad_kernel<<<grid_features, block1d>>>(d_grad, d_grad_temp, beta, lambda_reg, inv_n_samples, n_features);
        cudaDeviceSynchronize();

        // w = p * (1 - p)
        compute_w_kernel<<<grid_samples, block1d>>>(d_p, d_w, n_samples);
        cudaDeviceSynchronize();

        // temp = diag(w) * X
        diag_mul_kernel<<<grid_temp, block2d>>>(d_w, X, d_temp, n_samples, n_features);
        cudaDeviceSynchronize();

        // Hessian = X^T * temp / n_samples
        cudaMemset(d_Hessian, 0, n_features * n_features * sizeof(float));
        matrix_multiply_kernel<<<grid_hessian, block2d>>>(
            X, d_temp, d_Hessian, 
            n_samples, n_features, n_features,
            inv_n_samples
        );
        cudaDeviceSynchronize();

        // Add regularization: Hessian += (lambda_reg + epsilon) * I
        add_regularization_kernel<<<(n_features + 255) / 256, 256>>>(
            d_Hessian, n_features, lambda_reg + epsilon
        );
        cudaDeviceSynchronize();

        // Cholesky decomposition: Hessian = L * L^T
        cudaMemset(d_L, 0, n_features * n_features * sizeof(float));
        cholesky_decomp_kernel<<<1, 1>>>(d_Hessian, d_L, n_features);
        cudaDeviceSynchronize();

        // Solve L * y_temp = grad (forward)
        solve_forward_kernel<<<1, 1>>>(d_L, d_grad, d_y_temp, n_features);
        cudaDeviceSynchronize();

        // Solve L^T * delta = y_temp (backward)
        solve_backward_kernel<<<1, 1>>>(d_L, d_y_temp, d_delta, n_features);
        cudaDeviceSynchronize();

        // beta -= delta
        update_beta_kernel<<<grid_features, block1d>>>(beta, d_delta, n_features);
        cudaDeviceSynchronize();
    }

    cudaFree(d_z);
    cudaFree(d_p);
    cudaFree(d_w);
    cudaFree(d_diff);
    cudaFree(d_grad);
    cudaFree(d_grad_temp);
    cudaFree(d_Hessian);
    cudaFree(d_L);
    cudaFree(d_y_temp);
    cudaFree(d_delta);
    cudaFree(d_temp);
}