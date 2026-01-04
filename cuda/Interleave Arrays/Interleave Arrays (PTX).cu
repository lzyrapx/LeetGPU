#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ output,
                                         int N) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = t * 2;
    
    if (idx + 1 < N) {
        float2 a2, b2;
        float4 out4;
        
        // PTX
        asm volatile(
            "ld.global.v2.f32 {%0, %1}, [%4];\n"
            "ld.global.v2.f32 {%2, %3}, [%5];\n"
            "st.global.v4.f32 [%6], {%0, %2, %1, %3};"
            : "=f"(out4.x), "=f"(out4.z), "=f"(out4.y), "=f"(out4.w)
            : "l"(A + idx), "l"(B + idx), "l"(output + idx * 2)
        );
    } else if (idx < N) {
        output[2 * idx] = A[idx];
        output[2 * idx + 1] = B[idx];
    }
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
