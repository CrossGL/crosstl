#include <metal_stdlib>
using namespace metal;
// Hazard H15: float to half conversion rounds to nearest, ties to even, and overflows to infinity.
kernel void hazard(device const float* a [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    o[i] = float(half(a[i]));
}
