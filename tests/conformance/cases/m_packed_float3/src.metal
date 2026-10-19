#include <metal_stdlib>
using namespace metal;
// Hazard H16: packed_float3 has a 12-byte stride (float3 would be 16).
kernel void hazard(device const packed_float3* v [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    o[i] = v[i].y;
}
