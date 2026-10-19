#include <metal_stdlib>
using namespace metal;
// Hazard H6: fmin returns the non-NaN operand (MSL; C99 fmin).
kernel void hazard(device const float* xy [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    o[i] = fmin(xy[2 * i], xy[2 * i + 1]);
}
