#include <metal_stdlib>
using namespace metal;
// Hazard: fmod takes the sign of the dividend (C semantics): fmod(-5.5, 2) = -1.5.
kernel void hazard(device const float* xy [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    o[i] = fmod(xy[2 * i], xy[2 * i + 1]);
}
