#include <metal_stdlib>
using namespace metal;

// MSL (safe math): isnan(x) is true exactly for NaN.
kernel void hazard(device const float *inp [[buffer(0)]], device int *outp [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    outp[i] = isnan(inp[i]) ? 1 : 0;
}
