#include <metal_stdlib>
using namespace metal;

// MSL 4.1 section 3.1 item 11: the shift count is masked to its log2(N) low bits.
kernel void hazard(device const uint *inp [[buffer(0)]], device uint *outp [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    uint x = inp[2 * i];
    uint s = inp[2 * i + 1];
    outp[i] = (x << s) ^ (x >> s);
}
