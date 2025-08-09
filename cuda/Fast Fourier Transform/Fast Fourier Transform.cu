#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ unsigned int bitReverse(unsigned int x, int num_bits) {
    unsigned int rev = 0;
    for (int i = 0; i < num_bits; i++) {
        rev <<= 1;
        rev |= (x >> i) & 1;
    }
    return rev;
}

__global__ void kernel_bit_reverse(const float* in, float* out, int N, int num_bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    unsigned int rev_idx = bitReverse(idx, num_bits);
    out[2 * rev_idx] = in[2 * idx];
    out[2 * rev_idx + 1] = in[2 * idx + 1];
}

__global__ void kernel_butterfly(float* in, float* out, int step, int s, int N, bool inverse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_butterflies = N / 2;
    if (idx >= total_butterflies) return;

    int block_id = idx / step;
    int butterfly_idx = idx % step;

    int i = block_id * s + butterfly_idx;
    int j = i + step;

    float angle = -2.0f * M_PI * (float)butterfly_idx / (float)s;
    if (inverse) 
        angle = -angle;

    float w_real = cosf(angle);
    float w_imag = sinf(angle);

    float a_real = in[2 * i];
    float a_imag = in[2 * i + 1];
    float b_real = in[2 * j];
    float b_imag = in[2 * j + 1];

    float bw_real = b_real * w_real - b_imag * w_imag;
    float bw_imag = b_real * w_imag + b_imag * w_real;

    out[2 * i] = a_real + bw_real;
    out[2 * i + 1] = a_imag + bw_imag;
    out[2 * j] = a_real - bw_real;
    out[2 * j + 1] = a_imag - bw_imag;
}

__global__ void kernel_scale(float* in, float* out, int N, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[2 * idx] = in[2 * idx] * scale;
    out[2 * idx + 1] = in[2 * idx + 1] * scale;
}

void fft_device(const float* d_in, float* d_out, int N, bool inverse) {
    int num_bits = 0;
    int temp = N;
    while (temp > 1) {
        num_bits++;
        temp >>= 1;
    }

    float *d_buf1, *d_buf2;
    cudaMalloc(&d_buf1, 2 * N * sizeof(float));
    cudaMalloc(&d_buf2, 2 * N * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    kernel_bit_reverse<<<blocks, threads>>>(d_in, d_buf1, N, num_bits);

    float *d_in_cur = d_buf1;
    float *d_out_cur = d_buf2;

    int step = 1;
    int s = 2;
    while (s <= N) {
        int num_butterflies = N / 2;
        blocks = (num_butterflies + threads - 1) / threads;
        kernel_butterfly<<<blocks, threads>>>(d_in_cur, d_out_cur, step, s, N, inverse);
        float *temp_ptr = d_in_cur;
        d_in_cur = d_out_cur;
        d_out_cur = temp_ptr;
        step = s;
        s *= 2;
    }

    if (inverse) {
        kernel_scale<<<blocks, threads>>>(d_in_cur, d_out, N, 1.0f / N);
    } else {
        cudaMemcpy(d_out, d_in_cur, 2 * N * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_buf1);
    cudaFree(d_buf2);
}

__global__ void kernel_bluestein_a(const float* signal, float* a, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float x_real = signal[2 * n];
    float x_imag = signal[2 * n + 1];
    float angle = -M_PI * n * n / N;
    float w_real = cosf(angle);
    float w_imag = sinf(angle);
    a[2 * n] = x_real * w_real - x_imag * w_imag;
    a[2 * n + 1] = x_real * w_imag + x_imag * w_real;
}

__global__ void kernel_bluestein_d(float* d, int M, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= M) return;
    if (k == 0) {
        d[0] = 1.0f;
        d[1] = 0.0f;
    } else if (k < N) {
        float angle = M_PI * k * k / N;
        d[2 * k] = cosf(angle);
        d[2 * k + 1] = sinf(angle);
    } else if (k >= M - (N - 1)) {
        int idx = M - k;
        float angle = M_PI * idx * idx / N;
        d[2 * k] = cosf(angle);
        d[2 * k + 1] = sinf(angle);
    } else {
        d[2 * k] = 0.0f;
        d[2 * k + 1] = 0.0f;
    }
}

__global__ void kernel_complex_multiply(const float* a, const float* b, float* c, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    float a_real = a[2 * idx];
    float a_imag = a[2 * idx + 1];
    float b_real = b[2 * idx];
    float b_imag = b[2 * idx + 1];
    c[2 * idx] = a_real * b_real - a_imag * b_imag;
    c[2 * idx + 1] = a_real * b_imag + a_imag * b_real;
}

__global__ void kernel_final_multiply(const float* conv, float* spectrum, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    float conv_real = conv[2 * k];
    float conv_imag = conv[2 * k + 1];
    float angle = -M_PI * k * k / N;
    float w_real = cosf(angle);
    float w_imag = sinf(angle);
    spectrum[2 * k] = conv_real * w_real - conv_imag * w_imag;
    spectrum[2 * k + 1] = conv_real * w_imag + conv_imag * w_real;
}

unsigned int nextPowerOfTwo(unsigned int x) {
    if (x <= 1) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

bool isPowerOfTwo(int n) {
    if (n <= 0) return false;
    return (n & (n - 1)) == 0;
}

extern "C" void solve(const float* signal, float* spectrum, int N) {
    if (isPowerOfTwo(N)) {
        // for N is power of 2: Cooley-Tukey algorithm
        fft_device(signal, spectrum, N, false);
    } else {
        // for any N: Bluestein algorithm
        unsigned int M = nextPowerOfTwo(2 * N - 1);

        float *d_a, *d_d, *d_A, *d_B, *d_C, *d_conv;
        cudaMalloc(&d_a, 2 * M * sizeof(float));
        cudaMalloc(&d_d, 2 * M * sizeof(float));
        cudaMalloc(&d_A, 2 * M * sizeof(float));
        cudaMalloc(&d_B, 2 * M * sizeof(float));
        cudaMalloc(&d_C, 2 * M * sizeof(float));
        cudaMalloc(&d_conv, 2 * M * sizeof(float));

        cudaMemset(d_a, 0, 2 * M * sizeof(float));
        cudaMemset(d_d, 0, 2 * M * sizeof(float));

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        kernel_bluestein_a<<<blocks, threads>>>(signal, d_a, N);

        blocks = (M + threads - 1) / threads;
        kernel_bluestein_d<<<blocks, threads>>>(d_d, M, N);

        fft_device(d_a, d_A, M, false);
        fft_device(d_d, d_B, M, false);

        kernel_complex_multiply<<<blocks, threads>>>(d_A, d_B, d_C, M);

        fft_device(d_C, d_conv, M, true);

        blocks = (N + threads - 1) / threads;
        kernel_final_multiply<<<blocks, threads>>>(d_conv, spectrum, N);

        cudaFree(d_a);
        cudaFree(d_d);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_conv);
    }
}