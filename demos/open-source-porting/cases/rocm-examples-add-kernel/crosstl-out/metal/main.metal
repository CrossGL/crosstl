
#include <metal_stdlib>
using namespace metal;

// Compute Shader
kernel void AddKernel(device float* a [[buffer(0)]], const device float* b [[buffer(1)]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
    int global_idx = thread_position_in_threadgroup.x + threadgroup_position_in_grid.x * threads_per_threadgroup.x;
    a[global_idx] += b[global_idx];
}

