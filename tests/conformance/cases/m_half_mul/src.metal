#include <metal_stdlib>
using namespace metal;
// Hazard H15: half arithmetic rounds every result to binary16.
kernel void hazard(device const float* a [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    half h = half(a[i]);
    h = h * half(3.0h);
    o[i] = float(h);
}
