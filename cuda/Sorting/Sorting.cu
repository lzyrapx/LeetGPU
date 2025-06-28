// https://leetgpu.com/challenges/sorting

#include "solve.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

void solve(float* data, int N) {
    thrust::device_ptr<float> dev_ptr(data);
    thrust::sort(dev_ptr, dev_ptr + N);
}