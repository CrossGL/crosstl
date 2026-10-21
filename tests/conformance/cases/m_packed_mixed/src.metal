#include <metal_stdlib>
using namespace metal;
// A struct mixing float3 and packed_float3 members: a is 16 bytes at 0, s at 16, the packed b 12 bytes at 20, c at
// 32 and d at 40, and the 16-byte alignment of a pads the element to 48 bytes. The HLSL structured-buffer rules
// put s at 12 and b at 16, and std430 b at 16 as well.
struct Q {
    float3 a;
    float s;
    packed_float3 b;
    float c;
    float2 d;
};
kernel void hazard(device const Q* src [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    Q q = src[i];
    o[6 * i + 0] = q.a.z;
    o[6 * i + 1] = q.b.x;
    o[6 * i + 2] = q.b.z + 0.5;
    o[6 * i + 3] = q.c;
    o[6 * i + 4] = q.d.y;
    o[6 * i + 5] = q.a.x + q.d.x + q.s;
}
