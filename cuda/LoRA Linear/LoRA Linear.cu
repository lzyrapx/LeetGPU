#include <cuda_runtime.h>

// 转置函数
// 将 src (rows x cols) 转置为 dst (cols x rows)
__global__ void transpose_kernel(const float* src, float* dst, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        dst[x * rows + y] = src[y * cols + x];
    }
}

// GEMM 函数 
// C = alpha * A * B + (accum ? C : 0)
// A: M x K, B: K x N, C: M x N (行优先)
__global__ void gemm_kernel(const float* A, const float* B, float* C,
                            int M, int N, int K, float alpha, bool accum) {
    const int BLOCK_M = 32;
    const int BLOCK_N = 32;
    const int BLOCK_K = 32;

    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Bs[BLOCK_K][BLOCK_N];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_M + ty;
    int col = bx * BLOCK_N + tx;

    float sum = 0.0f;

    for (int k = 0; k < K; k += BLOCK_K) {
        // 加载 A 的分块 (BLOCK_M x BLOCK_K)
        if (row < M && k + tx < K) {
            As[ty][tx] = A[row * K + (k + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        // 加载 B 的分块 (BLOCK_K x BLOCK_N)
        if (col < N && k + ty < K) {
            Bs[ty][tx] = B[(k + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();

        // 计算部分内积
        #pragma unroll
        for (int i = 0; i < BLOCK_K; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        if (accum) {
            C[row * N + col] += alpha * sum;
        } else {
            C[row * N + col] = alpha * sum;
        }
    }
}

// x, W, A, B, output are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B, float* output,
                      int batch, int d_in, int d_out, int rank, float lora_scale) {
    // 三个转置矩阵
    float *Wt, *At, *Bt;
    cudaMalloc(&Wt, d_in * d_out * sizeof(float));
    cudaMalloc(&At, d_in * rank * sizeof(float));
    cudaMalloc(&Bt, rank * d_out * sizeof(float));

    // 转置 W (d_out x d_in) -> Wt (d_in x d_out)
    dim3 block(32, 32);
    dim3 grid((d_in + block.x - 1) / block.x, (d_out + block.y - 1) / block.y);
    transpose_kernel<<<grid, block>>>(W, Wt, d_out, d_in);

    // 转置 A (rank x d_in) -> At (d_in x rank)
    grid = dim3((d_in + block.x - 1) / block.x, (rank + block.y - 1) / block.y);
    transpose_kernel<<<grid, block>>>(A, At, rank, d_in);

    // 转置 B (d_out x rank) -> Bt (rank x d_out)
    grid = dim3((rank + block.x - 1) / block.x, (d_out + block.y - 1) / block.y);
    transpose_kernel<<<grid, block>>>(B, Bt, d_out, rank);

    cudaDeviceSynchronize();

    // 计算 output = x * W^T  (使用 Wt)
    int M = batch, N = d_out, K = d_in;
    dim3 gemm_block(32, 32);
    dim3 gemm_grid((N + gemm_block.x - 1) / gemm_block.x,
                   (M + gemm_block.y - 1) / gemm_block.y);
    gemm_kernel<<<gemm_grid, gemm_block>>>(x, Wt, output, M, N, K, 1.0f, false);

    // 临时矩阵 temp (batch x rank)
    float* temp;
    cudaMalloc(&temp, batch * rank * sizeof(float));

    // 计算 temp = x * A^T (使用 At)
    N = rank;
    gemm_grid = dim3((N + gemm_block.x - 1) / gemm_block.x,
                     (M + gemm_block.y - 1) / gemm_block.y);
    gemm_kernel<<<gemm_grid, gemm_block>>>(x, At, temp, M, N, K, 1.0f, false);

    // 计算 LoRA 部分并累加到 output: output += lora_scale * (temp * B^T) (使用 Bt)
    M = batch; N = d_out; K = rank;
    gemm_grid = dim3((N + gemm_block.x - 1) / gemm_block.x,
                     (M + gemm_block.y - 1) / gemm_block.y);
    gemm_kernel<<<gemm_grid, gemm_block>>>(temp, Bt, output, M, N, K, lora_scale, true);

    cudaDeviceSynchronize();

    cudaFree(Wt);
    cudaFree(At);
    cudaFree(Bt);
    cudaFree(temp);
}
