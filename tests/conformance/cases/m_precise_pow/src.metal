#include <metal_stdlib>
using namespace metal;
// Hazard H8: precise::pow keeps its 16 ulp bound for large and small results, whatever the target's math mode.
kernel void hazard(device const float* xy [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    o[i] = precise::pow(xy[2 * i], xy[2 * i + 1]);
}
