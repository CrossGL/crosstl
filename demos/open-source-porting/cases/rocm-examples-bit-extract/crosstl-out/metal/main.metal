
#include <metal_stdlib>
using namespace metal;

// Compute Shader
kernel void bit_extract_kernel(device uint* d_output [[buffer(0)]], const device uint* d_input [[buffer(1)]], constant uint& size [[buffer(2)]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint3 threads_per_threadgroup [[threads_per_threadgroup]], uint3 threadgroups_per_grid [[threadgroups_per_grid]]) {
    uint offset = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
    uint stride = threads_per_threadgroup.x * threadgroups_per_grid.x;
    for (uint i = offset; i < size; i += stride) {
        d_output[i] = d_input[i] >> 8 & 15u;
    }
}
