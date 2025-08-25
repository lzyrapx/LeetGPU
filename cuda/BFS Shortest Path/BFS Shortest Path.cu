#include <cuda_runtime.h>
#include <algorithm>

__global__ void bfs_kernel(const int* input_grid, int* distances, int rows, int cols, 
                          const int* current_frontier, int current_size, 
                          int* next_frontier, int* next_size, int current_level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current_size) return;

    int node = current_frontier[idx];
    int row = node / cols;
    int col = node % cols;

    // direction
    int dr[] = {1, -1, 0, 0};
    int dc[] = {0, 0, 1, -1};

    for (int d = 0; d < 4; d++) {
        int r = row + dr[d];
        int c = col + dc[d];
        if (r >= 0 && r < rows && c >= 0 && c < cols) {
            int index = r * cols + c;
            if (input_grid[index] == 0) {  // free ceil
                int expected = -1;
                // atomic compare and swap
                // if distances[index] == -1, then expected = current_level + 1 and will return distances[index]'s old value
                int old = atomicCAS(&distances[index], expected, current_level + 1);
                if (old == -1) {
                    // same as:
                    // pos = next_size
                    // next_size += 1
                    int pos = atomicAdd(next_size, 1);
                    next_frontier[pos] = index;
                }
            }
        }
    }
}

// grid, result are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, 
                     int start_row, int start_col, int end_row, int end_col) {
    if (start_row == end_row && start_col == end_col) {
        int zero = 0;
        cudaMemcpy(result, &zero, sizeof(int), cudaMemcpyHostToDevice);
        return;
    }

    int* distances = nullptr;
    int* frontier1 = nullptr;
    int* frontier2 = nullptr;
    int* d_next_size = nullptr;

    cudaMalloc(&distances, rows * cols * sizeof(int));
    cudaMemset(distances, -1, rows * cols * sizeof(int));

    cudaMalloc(&frontier1, rows * cols * sizeof(int));  // current frontier
    cudaMalloc(&frontier2, rows * cols * sizeof(int));  // next frontier
    cudaMalloc(&d_next_size, sizeof(int));

    int start_index = start_row * cols + start_col;
    cudaMemset(&distances[start_index], 0, sizeof(int));

    cudaMemcpy(frontier1, &start_index, sizeof(int), cudaMemcpyHostToDevice);

    int current_size = 1;
    int* current_frontier = frontier1;
    int* next_frontier = frontier2;
    int level = 0;

    int end_index = end_row * cols + end_col;
    int found = 0;
    int answer = -1;

    while (current_size > 0) {
        // reset next size
        cudaMemset(d_next_size, 0, sizeof(int));

        dim3 block(256);
        dim3 grid_dim((current_size + 255) / 256);
        bfs_kernel<<<grid_dim, block>>>(grid, distances, rows, cols, current_frontier, current_size, next_frontier, d_next_size, level);
        cudaDeviceSynchronize();

        // check if found
        int end_distance;
        cudaMemcpy(&end_distance, &distances[end_index], sizeof(int), cudaMemcpyDeviceToHost);
        if (end_distance != -1) {
            answer = end_distance;
            found = 1;
            break;
        }
        
        int next_size_val;
        cudaMemcpy(&next_size_val, d_next_size, sizeof(int), cudaMemcpyDeviceToHost);
        if (next_size_val == 0) {
            // no free ceil
            break;
        }
        // next frontier become current frontier
        std::swap(current_frontier, next_frontier);
        current_size = next_size_val;
        level++;
    }

    if (found) {
        cudaMemcpy(result, &answer, sizeof(int), cudaMemcpyHostToDevice);
    } else {
        int minus_one = -1;
        cudaMemcpy(result, &minus_one, sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaFree(distances);
    cudaFree(frontier1);
    cudaFree(frontier2);
    cudaFree(d_next_size);
}