// https://leetgpu.com/challenges/multi-agent-simulation

#include <cuda_runtime.h>

__global__ void update_agents_kernel(const float* agents, float* agents_next, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x_i = agents[4 * i];
    float y_i = agents[4 * i + 1];
    float vx_i = agents[4 * i + 2];
    float vy_i = agents[4 * i + 3];

    float sum_vx = 0.0f;
    float sum_vy = 0.0f;
    int count = 0;

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        float x_j = agents[4 * j];
        float y_j = agents[4 * j + 1];
        float dx = x_i - x_j;
        float dy = y_i - y_j;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq < 25.0f) {  // Check if distance < 5.0 (r^2 = 25)
            sum_vx += agents[4 * j + 2];
            sum_vy += agents[4 * j + 3];
            ++count;
        }
    }

    float new_vx, new_vy;
    if (count > 0) {
        float avg_vx = sum_vx / count;
        float avg_vy = sum_vy / count;
        new_vx = vx_i + 0.05f * (avg_vx - vx_i);
        new_vy = vy_i + 0.05f * (avg_vy - vy_i);
    } else {
        new_vx = vx_i;
        new_vy = vy_i;
    }

    agents_next[4 * i] = x_i + new_vx;      // new_x
    agents_next[4 * i + 1] = y_i + new_vy;  // new_y
    agents_next[4 * i + 2] = new_vx;        // new_vx
    agents_next[4 * i + 3] = new_vy;        // new_vy
}

// agents, agents_next are device pointers
extern "C" void solve(const float* agents, float* agents_next, int N) {
    float *d_agents, *d_agents_next;
    cudaMalloc(&d_agents, 4 * N * sizeof(float));
    cudaMalloc(&d_agents_next, 4 * N * sizeof(float));

    cudaMemcpy(d_agents, agents, 4 * N * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    update_agents_kernel<<<grid_size, block_size>>>(d_agents, d_agents_next, N);

    cudaMemcpy(agents_next, d_agents_next, 4 * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_agents);
    cudaFree(d_agents_next);
}