#include <metal_stdlib>
using namespace metal;

constant uint multiplier [[function_constant(7)]] = 3u;

kernel void specialized_output(
    device const uint* input_values [[buffer(0)]],
    device uint* output_values [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    output_values[index] = input_values[index] * multiplier;
}
