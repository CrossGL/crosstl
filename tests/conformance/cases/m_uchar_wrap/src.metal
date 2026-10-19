#include <metal_stdlib>
using namespace metal;
// Hazard N1: 8-bit unsigned arithmetic wraps modulo 256 when stored back to uchar (C++ promotion + truncation).
kernel void hazard(device const int* a [[buffer(0)]],
                   device int* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    uchar v = uchar(a[i]);
    v = v + uchar(1);
    o[i] = int(v);
}
