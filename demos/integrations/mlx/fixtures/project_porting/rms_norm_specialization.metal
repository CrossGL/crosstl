#include <metal_stdlib>
using namespace metal;

constant bool has_w [[function_constant(20)]];

kernel void vjp_rmsfloat32(
    device float* output [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = has_w ? 1.0f : 0.0f;
}
