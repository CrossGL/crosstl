#include <metal_stdlib>
using namespace metal;

// Pointer arithmetic on a device pointer (C++ rules).
kernel void hazard(device const int *inp [[buffer(0)]], device int *outp [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    device const int *p = inp + 3;
    outp[i] = p[i];
}
