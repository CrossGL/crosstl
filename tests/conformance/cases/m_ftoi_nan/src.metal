#include <metal_stdlib>
using namespace metal;

// MSL 4.1 section 8.6: converting a NaN to an integer type yields 0 (inputs are NaN or in range).
kernel void hazard(device const float *inp [[buffer(0)]], device int *outp [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    float x = inp[i];
    outp[i] = int(x);
}
