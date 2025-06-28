// https://leetgpu.com/challenges/prefix-sum

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>

// prefix sum for small array (lunch with just one block)
__global__ void prefixSum_UniqueBlock(float* in, int in_length, float* out ){

	// shared memory declaration
	extern __shared__ float DSM[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// load in shared memory
	if(idx < in_length){
		DSM[threadIdx.x] = in[idx];

		// partial sums phase
		for(int stride = 1; stride <= blockDim.x; stride *= 2){
			__syncthreads();
			int index_aux = (threadIdx.x + 1) * 2 * stride - 1;
			if(index_aux < blockDim.x)
				DSM[index_aux] += DSM[index_aux - stride];
		}

		// reduction phase
		for(int stride=blockDim.x/4 ; stride > 0 ; stride /= 2){
			__syncthreads();

			int index_aux = (threadIdx.x + 1) * 2 * stride - 1;
			if(index_aux + stride < blockDim.x)
				DSM[index_aux + stride] += DSM[index_aux];
		}

		__syncthreads();

		out[idx] = DSM[threadIdx.x];

	}

}

// prefix sum for multiple blocks
__global__ void prefixSum_multiBlocks(float* in, int in_length, float* out, float* temp ){
	extern __shared__ float DSM[];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// load in shared memory
	if(idx < in_length){
		DSM[threadIdx.x] = in[idx];

		// partial sums phase
		for(int stride = 1; stride <= blockDim.x; stride *= 2){
			__syncthreads();
			int index_aux = (threadIdx.x + 1) * 2 * stride - 1;
			if(index_aux < blockDim.x)
				DSM[index_aux] += DSM[index_aux - stride];
		}

		// reduction phase
		for(int stride=blockDim.x/4 ; stride > 0 ; stride /= 2){
			__syncthreads();

			int index_aux = (threadIdx.x + 1) * 2 * stride - 1;
			if(index_aux + stride < blockDim.x)
				DSM[index_aux + stride] += DSM[index_aux];
		}

		__syncthreads();

		// save intermediary values on temp to post combine for multi blocks
		if(threadIdx.x == 0)
			temp[blockIdx.x] = DSM[blockDim.x - 1];

		out[idx] = DSM[threadIdx.x];

	}

}

// combine for multiple blocks
__global__ void prefixsum_combine(float* in, int in_length, float* out, int out_length) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < out_length && blockIdx.x > 0){
		out[idx] += in[blockIdx.x - 1];
	}
}

void solve(const float* input, float* output, int N) {
    // declaring variables
    float blockSize = 256;
    float *dev_in, *dev_out, *dev_temp;
    int size = N * sizeof(float);
    int tempSize = blockSize * sizeof(float);

    // allocate memory
    cudaMalloc(&dev_in, size);
    cudaMalloc(&dev_out, size);
    cudaMalloc(&dev_temp, tempSize);

    // copy data to device
    cudaMemcpy(dev_in, input, size, cudaMemcpyHostToDevice);

    // launch kernel to compute prefix sum for each block and save partial block sum on dev_temp
    dim3 blocks(blockSize, 1, 1);
    dim3 grid(ceil((float)N / blockSize), 1, 1);  // Added float cast for proper ceiling
    int shMem = blockSize * sizeof(float);
    prefixSum_multiBlocks<<<grid, blocks, shMem>>>(dev_in, N, dev_out, dev_temp);
    cudaDeviceSynchronize();

    // launch kernel to compute prefix sum on each block total cell stores in dev_temp
    int topBlocksQtd = ceil((float)N / blockSize);  // Added float cast for proper ceiling
    prefixSum_UniqueBlock<<<1, topBlocksQtd, topBlocksQtd * sizeof(float)>>>(dev_temp, topBlocksQtd, dev_temp);
    cudaDeviceSynchronize();

    // launch kernel to combine blocks results
    prefixsum_combine<<<grid, blocks>>>(dev_temp, topBlocksQtd, dev_out, N);
    cudaDeviceSynchronize();

    // copy data to host
    cudaMemcpy(output, dev_out, size, cudaMemcpyDeviceToHost);

    // free memory space
    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaFree(dev_temp);
}