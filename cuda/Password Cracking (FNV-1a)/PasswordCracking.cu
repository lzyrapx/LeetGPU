// https://leetgpu.com/challenges/password-cracking-fnv-1a

#include "solve.h"
#include <cuda_runtime.h>

// FNV-1a hash function that takes a byte array and its length as input
// Returns a 32-bit unsigned integer hash value
__device__ unsigned int fnv1a_hash_bytes(const unsigned char* data, int length) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    for (int i = 0; i < length; i++) {
        hash = (hash ^ data[i]) * FNV_PRIME;
    }
    return hash;
}

__device__ unsigned int repeat_hash(const unsigned char* password, int password_length, int R) {
    unsigned int hash = fnv1a_hash_bytes(password, password_length);
    #pragma unroll
    for (int i = 0; i < R - 1; i++) {
        hash = fnv1a_hash_bytes((const unsigned char*)&hash, sizeof(hash));
    }
    return hash;
}

template<int password_length>
__global__ void password_crack_kernel(unsigned int target_hash, int R, char* output_password) {
    int tid = threadIdx.x;
    char ans[password_length + 1];
    ans[password_length] = '\0';
    ans[0] = 'a' + (tid % 26);
    if constexpr (password_length > 1) {
        ans[1] = 'a' + (tid / 26);
    }
    if constexpr (password_length > 2) {
        int block_id = blockIdx.x;
        #pragma unroll
        for (int i = 2; i < password_length; i++) {
            ans[i] = 'a' + (block_id % 26);
            block_id /= 26;
        }
    }
    unsigned int hash = repeat_hash((const unsigned char*)ans, password_length, R);
    if (hash == target_hash) {
        for (int i = 0; i < password_length; i++) {
            output_password[i] = ans[i];
        }
        output_password[password_length] = '\0';
    }
}

constexpr int grid_sizes[] = {0, 1, 1, 26, 26*26, 26*26*26, 26*26*26*26};
constexpr int block_sizes[] = {0, 26, 26*26, 26*26, 26*26, 26*26, 26*26};

#define HANDLE_CASE(n) \
    case n: \
        password_crack_kernel<n><<<grid_sizes[n], block_sizes[n]>>>(target_hash, R, output_password); \
        break;

void solve(unsigned int target_hash, int password_length, int R, char* output_password) {
  switch (password_length) {
      HANDLE_CASE(1)
      HANDLE_CASE(2)
      HANDLE_CASE(3)
      HANDLE_CASE(4)
      HANDLE_CASE(5)
      HANDLE_CASE(6)
      default:
          return;
  }
  }