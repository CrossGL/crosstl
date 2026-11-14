#include <metal_stdlib>
using namespace metal;
// clz and ctz (MSL 3.2 section 6.13) of uint, int and vector components; a zero component counts 32.
kernel void bit_scans(device const uint* a [[buffer(0)]], device uint* o [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    uint x = a[i];
    int s = as_type<int>(x);
    uint2 pair = clz(uint2(x, ~x));
    int3 counts = ctz(int3(s, ~s, as_type<int>(x >> 16u)));
    o[8u * i] = clz(x);
    o[8u * i + 1u] = metal::ctz(x);
    o[8u * i + 2u] = uint(::metal::clz(s));
    o[8u * i + 3u] = uint(ctz(s));
    o[8u * i + 4u] = pair.x;
    o[8u * i + 5u] = pair.y;
    o[8u * i + 6u] = uint(counts.y);
    o[8u * i + 7u] = uint(counts.z);
}
