#include <metal_stdlib>
using namespace metal;

// dispatchThreads launches exactly N threads (non-uniform threadgroups), so the kernel needs no guard.
kernel void hazard(device const int *inp [[buffer(0)]], device int *outp [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    outp[i] = int(i) * 3 + inp[0];
}
