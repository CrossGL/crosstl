#include <metal_stdlib>
using namespace metal;
// MSL float3x3 is three float3 columns, 16 bytes each: a 48-byte element whose column c starts at word 4c. An HLSL
// structured buffer of float3x3 has 36-byte elements.
kernel void hazard(device const float3x3* m [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    o[i] = m[i][2].x + (m[i] * float3(1.0, 2.0, 3.0)).y;
}
