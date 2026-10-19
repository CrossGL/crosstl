#include <metal_stdlib>
using namespace metal;

// MSL section 3.1 (C++ rules): the quotient truncates toward zero and (a/b)*b + a%b == a.
kernel void hazard(device const int *inp [[buffer(0)]], device int *outp [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    int a = inp[2 * i];
    int b = inp[2 * i + 1];
    outp[i] = a % b;
}
