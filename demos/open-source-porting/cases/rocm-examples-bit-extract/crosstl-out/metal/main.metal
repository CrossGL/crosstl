
#include <metal_stdlib>
using namespace metal;

// Compute Shader
kernel void bit_extract_kernel(device uint* d_output [[buffer(0)]], const device uint* d_input [[buffer(1)]], constant uint& size [[buffer(2)]], uint3 thread_position_in_grid [[thread_position_in_grid]], uint3 threads_per_threadgroup [[threads_per_threadgroup]], uint3 threadgroups_per_grid [[threadgroups_per_grid]]) {
    uint offset = thread_position_in_grid.x;
    uint stride = threads_per_threadgroup.x * threadgroups_per_grid.x;
    for (uint i = offset; i < size; i += stride) {
        d_output[i] = d_input[i] >> 8 & 15u;
    }
}
