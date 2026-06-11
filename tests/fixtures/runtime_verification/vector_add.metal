#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = lhs[id] + rhs[id];
}
