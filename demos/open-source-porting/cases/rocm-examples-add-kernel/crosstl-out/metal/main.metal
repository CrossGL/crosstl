
#include <metal_stdlib>
using namespace metal;

// Compute Shader
kernel void AddKernel(device float* a [[buffer(0)]], const device float* b [[buffer(1)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    int global_idx = thread_position_in_grid.x;
    a[global_idx] += b[global_idx];
}
