#include <metal_stdlib>
using namespace metal;
// Hazard H7/H8: precise::sin must stay accurate for large arguments, whatever the target's default math mode.
kernel void hazard(device const float* a [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    o[i] = precise::sin(a[i]);
}
