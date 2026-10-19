#include <metal_stdlib>
using namespace metal;
// Hazard H15: half sqrt rounds the exact root to binary16.
kernel void hazard(device const float* a [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    half x = half(a[i]);
    o[i] = float(sqrt(x));
}
