#include <metal_stdlib>
using namespace metal;
// popcount and reverse_bits (MSL 3.2 section 6.13) of 8- and 16-bit scalars and vectors take the value's own width.
kernel void narrow_bit_functions(device const uint* a [[buffer(0)]], device uint* o [[buffer(1)]],
                                 uint i [[thread_position_in_grid]]) {
    uint x = a[i];
    short s = short(x);
    ushort u = ushort(x >> 16u);
    short2 v = short2(s, short(x >> 4u));
    char c = char(x >> 8u);
    uchar4 b = uchar4(uchar(x), uchar(x >> 8u), uchar(x >> 16u), uchar(x >> 24u));
    short2 rv = reverse_bits(v);
    uchar4 rb = reverse_bits(b);
    o[8u * i] = uint(popcount(s));
    o[8u * i + 1u] = uint(ushort(reverse_bits(s)));
    o[8u * i + 2u] = uint(reverse_bits(u));
    o[8u * i + 3u] = uint(ushort(rv.y));
    o[8u * i + 4u] = uint(popcount(c));
    o[8u * i + 5u] = uint(rb.x) | (uint(rb.w) << 8u);
    o[8u * i + 6u] = uint(popcount(v).y);
    o[8u * i + 7u] = uint(popcount(b).z);
}
