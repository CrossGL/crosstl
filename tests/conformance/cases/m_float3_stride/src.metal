#include <metal_stdlib>
using namespace metal;
// Hazard H16: float3 is 16-byte sized and aligned, so a float3 buffer has a 16-byte stride (an HLSL structured
// buffer of float3 has 12).
kernel void hazard(device const float3* v [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    o[i] = v[i].y;
}
