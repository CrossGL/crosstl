#include <metal_stdlib>
using namespace metal;

// MSL 4.1 section 1.6.3: the pragma disables contraction, so a*b - c is not fused into fma(a, b, -c).
#pragma METAL fp contract(off)
kernel void hazard(device const float *inp [[buffer(0)]], device float *outp [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    float a = inp[3 * i];
    float b = inp[3 * i + 1];
    float c = inp[3 * i + 2];
    outp[i] = a * b - c;
}
