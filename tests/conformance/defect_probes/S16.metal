#include <metal_stdlib>
using namespace metal;
kernel void k1(device uint* a [[buffer(0)]], uint i [[thread_position_in_grid]]) { a[i] = a[i] + 1u; }
kernel void k2(device uint* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { b[i] = b[i] * 2u; }
