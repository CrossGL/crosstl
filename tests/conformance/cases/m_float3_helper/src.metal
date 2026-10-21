#include <metal_stdlib>
using namespace metal;
// Helpers that take buffer pointers read and write the caller's elements in the MSL natural layout: S is 32 bytes,
// with b at byte 16, and a pointer offset by one element moves 32 bytes.
struct S {
    float3 a;
    float b;
};
float sum(device const S* p, uint i) {
    return p[i].a.y + p[i].b;
}
void put(device S* q, uint i, float v) {
    q[i].b = v;
    q[i].a = float3(v, 0.0, 1.0);
}
kernel void hazard(device const S* src [[buffer(0)]], device S* dst [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    uint odd = i & 1u;
    put(dst + odd, i - odd, sum(src + odd, i - odd));
}
