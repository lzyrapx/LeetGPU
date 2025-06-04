// https://leetgpu.com/challenges/ordinary-least-squares-regression

#include "solve.h"
#include <cuda_runtime.h>
#include <cmath>

// 计算 X^T * X（Gram矩阵）
__global__ void xtx_kernel(const float* X, float* C, int n_samples, int n_features) {
    // 计算当前线程处理的矩阵元素坐标(i, j)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查坐标是否在有效范围内
    if (i < n_features && j < n_features) {
        float sum = 0.0f;
        // 计算 X 的第 i 列和 X 的第 j 列的点积
        for (int k = 0; k < n_samples; k++) {
            float xi = X[k * n_features + i];  // 第 k 行，第 i 列元素
            float xj = X[k * n_features + j];  // 第 k 行，第 j 列元素
            sum += xi * xj;  // 累加点积
        }
        // 将计算结果存储到 Gram 矩阵的(i,j)位置
        C[i * n_features + j] = sum;
    }
}

// 计算 X^T * y
__global__ void xty_kernel(const float* X, const float* y, float* b, int n_samples, int n_features) {
    // 计算当前线程处理的特征索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_features) {
        float sum = 0.0f;
        // 计算 X 的第 i 列与目标向量 y 的点积
        for (int k = 0; k < n_samples; k++) {
            sum += X[k * n_features + i] * y[k];  // 累加点积
        }
        // 将计算结果存储到输出向量的第 i 个位置
        b[i] = sum;
    }
}

// Cholesky 分解（将对称正定矩阵分解为 L * L^T）
// A: 输入/输出矩阵, n: 矩阵维度
__global__ void cholesky_kernel(float* L, int n) {
    // 当前线程ID（在一个线程块内）
    int tid = threadIdx.x;

    // 遍历矩阵的每一行
    for (int i = 0; i < n; i++) {
        // 处理对角线元素
        if (tid == i) {
            float sum = 0.0f;
            // 计算L[i][k]^2的累加和（k < i）
            for (int k = 0; k < i; k++) {
                float lik = L[i * n + k];
                sum += lik * lik;
            }
            // 计算对角线元素：L[i][i] = sqrt(L[i][i] - sum)
            L[i * n + i] = sqrtf(L[i * n + i] - sum);
        }
        __syncthreads();  // 同步线程块内所有线程

        // 处理下三角部分（i+1到n-1行）
        if (tid > i && tid < n) {
            float sum = 0.0f;
            // 计算L[tid][k] * L[i][k]的累加和（k < i）
            for (int k = 0; k < i; k++) {
                sum += L[tid * n + k] * L[i * n + k];
            }
            // 计算非对角线元素：L[tid][i] = (A[tid][i] - sum) / L[i][i]
            L[tid * n + i] = (L[tid * n + i] - sum) / L[i * n + i];
        }
        __syncthreads();  // 同步线程块内所有线程
    }
}

// 前向替换（求解下三角系统 L * w = b）
// L: 下三角矩阵, b: 输入/输出向量, n: 矩阵维度
__global__ void forward_substitution_kernel(const float* L, float* b, int n) {
    extern __shared__ float s_w[];
    int tid = threadIdx.x;  // 当前线程ID

    if (tid < n) {
        // 将全局内存中的向量 b 复制到共享内存s_w
        s_w[tid] = b[tid];
    }
    __syncthreads();  // 同步线程块内所有线程

    // 前向替换算法
    for (int i = 0; i < n; i++) {
        if (tid == i) {
            float sum = 0.0f;
            // 计算已知部分的累加和：sum = Σ(L[i][j] * w[j]), j < i
            for (int j = 0; j < i; j++) {
                sum += L[i * n + j] * s_w[j];
            }
            // 计算当前解：w[i] = (b[i] - sum) / L[i][i]
            s_w[i] = (s_w[i] - sum) / L[i * n + i];
        }
        __syncthreads();  // 同步线程块内所有线程
    }
    // 将结果从共享内存复制回全局内存
    if (tid < n) {
        b[tid] = s_w[tid];
    }
}

// 后向替换（求解上三角系统 L^T * beta = w）
// L: 下三角矩阵, w: 输入向量, beta: 输出向量, n: 矩阵维度
__global__ void backward_substitution_kernel(const float* L, const float* w, float* beta, int n) {
    extern __shared__ float s_beta[];
    int tid = threadIdx.x;  // 当前线程ID

    if (tid < n) {
        // 初始化共享内存中的 beta 向量为 0
        s_beta[tid] = 0.0f;
    }
    __syncthreads();  // 同步线程块内所有线程

    // 后向替换算法（从最后一行开始向前求解）
    for (int i = n - 1; i >= 0; i--) {
        if (tid == i) {
            float sum = 0.0f;
            // 计算已知部分的累加和：sum = Σ(L[j][i] * beta[j]), j > i
            for (int j = i + 1; j < n; j++) {
                // 注意：这里使用 L[j][i]是因为我们需要 L^T 的上三角部分
                sum += L[j * n + i] * s_beta[j];
            }
            // 计算当前解：beta[i] = (w[i] - sum) / L[i][i]
            s_beta[i] = (w[i] - sum) / L[i * n + i];
        }
        __syncthreads();
    }
    
    // 将最终结果从共享内存复制到输出向量
    if (tid < n) {
        beta[tid] = s_beta[tid];
    }
}

// X, y, beta are device pointers
// n_samples: 样本数量, n_features: 特征数量
void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    // 处理特征数为 0 的特殊情况
    if (n_features == 0) {
        return;
    }

    float *d_A;  // 存储 X^T * X (Gram矩阵)
    float *d_b;  // 存储 X^T * y

    cudaMalloc(&d_A, n_features * n_features * sizeof(float));
    cudaMalloc(&d_b, n_features * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((n_features + 15) / 16, (n_features + 15) / 16);

    // 计算 X^T * X
    xtx_kernel<<<grid, block>>>(X, d_A, n_samples, n_features);

    dim3 block1(256);
    dim3 grid1((n_features + 255) / 256);

    // 计算 X^T * y
    xty_kernel<<<grid1, block1>>>(X, y, d_b, n_samples, n_features);
    cudaDeviceSynchronize();

    // 对 Gram 矩阵进行 Cholesky 分解（使用一个线程块）, 参考：https://zhuanlan.zhihu.com/p/31100572316
    // 将 X^TX 分解成 LL^T
    cholesky_kernel<<<1, n_features>>>(d_A, n_features);
    cudaDeviceSynchronize();

    // 计算前向替换和后向替换所需的共享内存大小
    size_t shared_mem_size = n_features * sizeof(float);

    // 前向替换：求解 L * w = d_b
    forward_substitution_kernel<<<1, n_features, shared_mem_size>>>(d_A, d_b, n_features);
    cudaDeviceSynchronize();

    // 后向替换：求解 L^T * beta = w (结果直接存储到 beta)
    backward_substitution_kernel<<<1, n_features, shared_mem_size>>>(d_A, d_b, beta, n_features);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_b);
}
