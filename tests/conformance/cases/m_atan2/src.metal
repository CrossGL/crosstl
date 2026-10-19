#include <metal_stdlib>
using namespace metal;
// Hazard H10: precise::atan2 distinguishes +0 and -0 (atan2(-0, -1) = -pi).
kernel void hazard(device const float* yx [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    o[i] = precise::atan2(yx[2 * i], yx[2 * i + 1]);
}
