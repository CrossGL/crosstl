#include <metal_stdlib>
using namespace metal;
// MSL natural layout: S is 32 bytes, with b at byte 16 and c at 24; bytes 12-15 and 20-23 are padding. HLSL
// structured buffers and std430 put b at 12 and c at 16, in 24- and 32-byte elements.
struct S {
    float3 a;
    float b;
    float2 c;
};
kernel void hazard(device const S* src [[buffer(0)]], device S* dst [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    S s = src[i];
    S t;
    t.a = s.a.zyx + s.b;
    t.b = s.a.x - s.c.y;
    t.c = float2(s.b, s.a.y);
    dst[i] = t;
    dst[i].a.y += 0.5;
    dst[i].c.x *= 2.0;
}
