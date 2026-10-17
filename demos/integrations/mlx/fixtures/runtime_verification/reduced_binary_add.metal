#include <metal_stdlib>
using namespace metal;

kernel void mlx_binary_add_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    out[index] = lhs[index] + rhs[index];
}
