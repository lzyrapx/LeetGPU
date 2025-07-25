#include "solve.h"
#include <cuda_runtime.h>
#include <cmath>

#define MAX_BLOCK_SZ 128

static __global__ void sum_scan_blelloch_kernel(unsigned int* d_out, unsigned int* d_in, unsigned int len) {
    extern __shared__ unsigned int temp[];
    int thid = threadIdx.x;
    int offset = 1;

    temp[2*thid] = d_in[2*thid];
    temp[2*thid+1] = d_in[2*thid+1];

    for (int d = len >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0) {
        temp[len-1] = 0;
    }

    for (int d = 1; d < len; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    d_out[2*thid] = temp[2*thid];
    d_out[2*thid+1] = temp[2*thid+1];
}

static void sum_scan_blelloch(unsigned int* d_out, unsigned int* d_in, unsigned int len) {
    if (len == 0) return;
    if (len == 1) {
        cudaMemcpy(d_out, d_in, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        return;
    }
    
    unsigned int block_size = (len + 1) / 2;
    unsigned int shared_mem_size = 2 * block_size * sizeof(unsigned int);
    sum_scan_blelloch_kernel<<<1, block_size, shared_mem_size>>>(d_out, d_in, 2*block_size);
    cudaDeviceSynchronize();
}

__global__ void gpu_radix_sort_local(
    unsigned int* d_out_sorted,
    unsigned int* d_prefix_sums,
    unsigned int* d_block_sums,
    unsigned int input_shift_width,
    unsigned int* d_in,
    unsigned int d_in_len,
    unsigned int max_elems_per_block
) {
    extern __shared__ unsigned int shmem[];
    unsigned int* s_data = shmem;
    unsigned int s_mask_out_len = max_elems_per_block + 1;
    unsigned int* s_mask_out = &s_data[max_elems_per_block];
    unsigned int* s_merged_scan_mask_out = &s_mask_out[s_mask_out_len];
    unsigned int* s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];
    unsigned int* s_scan_mask_out_sums = &s_mask_out_sums[4];

    unsigned int thid = threadIdx.x;
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;

    if (cpy_idx < d_in_len)
        s_data[thid] = d_in[cpy_idx];
    else
        s_data[thid] = 0;
    __syncthreads();

    unsigned int t_data = s_data[thid];
    unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;

    for (unsigned int i = 0; i < 4; ++i) {
        s_mask_out[thid] = 0;
        if (thid == 0)
            s_mask_out[s_mask_out_len - 1] = 0;
        __syncthreads();

        bool val_equals_i = false;
        if (cpy_idx < d_in_len) {
            val_equals_i = t_2bit_extract == i;
            s_mask_out[thid] = val_equals_i;
        }
        __syncthreads();

        int partner = 0;
        unsigned int sum = 0;
        unsigned int max_steps = (unsigned int)log2f((float)max_elems_per_block);
        for (unsigned int d = 0; d < max_steps; d++) {
            partner = thid - (1 << d);
            if (partner >= 0) {
                sum = s_mask_out[thid] + s_mask_out[partner];
            }
            else {
                sum = s_mask_out[thid];
            }
            __syncthreads();
            s_mask_out[thid] = sum;
            __syncthreads();
        }

        unsigned int cpy_val = s_mask_out[thid];
        __syncthreads();
        s_mask_out[thid + 1] = cpy_val;
        __syncthreads();

        if (thid == 0) {
            s_mask_out[0] = 0;
            unsigned int total_sum = s_mask_out[s_mask_out_len - 1];
            s_mask_out_sums[i] = total_sum;
            d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
        }
        __syncthreads();

        if (val_equals_i && (cpy_idx < d_in_len)) {
            s_merged_scan_mask_out[thid] = s_mask_out[thid];
        }
        __syncthreads();
    }

    if (thid == 0) {
        unsigned int run_sum = 0;
        for (unsigned int i = 0; i < 4; ++i) {
            s_scan_mask_out_sums[i] = run_sum;
            run_sum += s_mask_out_sums[i];
        }
    }
    __syncthreads();

    if (cpy_idx < d_in_len) {
        unsigned int t_prefix_sum = s_merged_scan_mask_out[thid];
        unsigned int new_pos = t_prefix_sum + s_scan_mask_out_sums[t_2bit_extract];
        
        __syncthreads();

        s_data[new_pos] = t_data;
        s_merged_scan_mask_out[new_pos] = t_prefix_sum;

        __syncthreads();

        d_prefix_sums[cpy_idx] = s_merged_scan_mask_out[thid];
        d_out_sorted[cpy_idx] = s_data[thid];
    }
}

__global__ void gpu_glbl_shuffle(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* d_scan_block_sums,
    unsigned int* d_prefix_sums,
    unsigned int input_shift_width,
    unsigned int d_in_len,
    unsigned int max_elems_per_block)
{
    unsigned int thid = threadIdx.x;
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;

    if (cpy_idx < d_in_len) {
        unsigned int t_data = d_in[cpy_idx];
        unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;
        unsigned int t_prefix_sum = d_prefix_sums[cpy_idx];
        unsigned int data_glbl_pos = d_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x] + t_prefix_sum;
        d_out[data_glbl_pos] = t_data;
    }
}

void radix_sort(unsigned int* const d_out,
    unsigned int* const d_in,
    unsigned int d_in_len)
{
    unsigned int block_sz = MAX_BLOCK_SZ;
    unsigned int max_elems_per_block = block_sz;
    unsigned int grid_sz = (d_in_len + max_elems_per_block - 1) / max_elems_per_block;

    unsigned int* d_prefix_sums;
    cudaMalloc(&d_prefix_sums, sizeof(unsigned int) * d_in_len);
    cudaMemset(d_prefix_sums, 0, sizeof(unsigned int) * d_in_len);

    unsigned int* d_block_sums;
    unsigned int d_block_sums_len = 4 * grid_sz;
    cudaMalloc(&d_block_sums, sizeof(unsigned int) * d_block_sums_len);
    cudaMemset(d_block_sums, 0, sizeof(unsigned int) * d_block_sums_len);

    unsigned int* d_scan_block_sums;
    cudaMalloc(&d_scan_block_sums, sizeof(unsigned int) * d_block_sums_len);
    cudaMemset(d_scan_block_sums, 0, sizeof(unsigned int) * d_block_sums_len);

    unsigned int s_data_len = max_elems_per_block;
    unsigned int s_mask_out_len = max_elems_per_block + 1;
    unsigned int s_merged_scan_mask_out_len = max_elems_per_block;
    unsigned int s_mask_out_sums_len = 4;
    unsigned int s_scan_mask_out_sums_len = 4;
    unsigned int shmem_sz = (s_data_len + s_mask_out_len + s_merged_scan_mask_out_len + s_mask_out_sums_len + s_scan_mask_out_sums_len) * sizeof(unsigned int);

    unsigned int* d_temp_in = d_in;
    unsigned int* d_temp_out = d_out;

    for (unsigned int shift_width = 0; shift_width <= 30; shift_width += 2) {
        gpu_radix_sort_local<<<grid_sz, block_sz, shmem_sz>>>(d_temp_out, d_prefix_sums, d_block_sums, shift_width, d_temp_in, d_in_len, max_elems_per_block);
        cudaDeviceSynchronize();

        sum_scan_blelloch(d_scan_block_sums, d_block_sums, d_block_sums_len);

        gpu_glbl_shuffle<<<grid_sz, block_sz>>>(d_temp_in, d_temp_out, d_scan_block_sums, d_prefix_sums, shift_width, d_in_len, max_elems_per_block);
        cudaDeviceSynchronize();

        std::swap(d_temp_in, d_temp_out);
    }

    if (d_temp_in != d_out) {
        cudaMemcpy(d_out, d_temp_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_scan_block_sums);
    cudaFree(d_block_sums);
    cudaFree(d_prefix_sums);
}

void solve(const unsigned int* input, unsigned int* output, int N) {
    if (N <= 0) return;
    
    unsigned int* d_temp;
    cudaMalloc(&d_temp, N * sizeof(unsigned int));
    cudaMemcpy(d_temp, input, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    
    radix_sort(output, d_temp, N);
    
    cudaFree(d_temp);
}