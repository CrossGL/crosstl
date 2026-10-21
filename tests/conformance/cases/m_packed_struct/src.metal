#include <metal_stdlib>
using namespace metal;
// MSL packed vectors are aligned to their scalar and not padded: P is 28 bytes, with b at byte 4, c at 16 and
// d at 20. float3 and float2 members would move b to 16, d to 24 and make the element 48 bytes.
struct P {
    float a;
    packed_float3 b;
    float c;
    packed_float2 d;
};
kernel void hazard(device const P* src [[buffer(0)]], device P* dst [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    P p = src[i];
    P q;
    q.a = p.d.y;
    q.b = p.b.zyx;
    q.c = p.a + p.c;
    q.d = float2(p.b.x, p.c);
    dst[i] = q;
    dst[i].b.y += 0.5;
}
