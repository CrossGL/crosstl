#include <metal_stdlib>
using namespace metal;
// Helpers that take buffer pointers read and write the caller's packed_float3 elements, which are 12 bytes apart;
// a pointer offset by one element moves 12 bytes.
float3 load3(device const packed_float3* p, uint i) {
    return float3(p[i]);
}
void store3(device packed_float3* q, uint i, float3 v) {
    q[i] = v;
    q[i].y += 0.5;
}
kernel void hazard(device const packed_float3* src [[buffer(0)]], device packed_float3* dst [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    uint odd = i & 1u;
    store3(dst + odd, i - odd, load3(src + odd, i - odd).zyx);
}
