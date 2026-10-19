#include <metal_stdlib>
using namespace metal;

// simd_sum reduces over the SIMD-group; its width is implementation-defined and reported to the host.
kernel void hazard(device const int *inp [[buffer(0)]], device int *outp [[buffer(1)]],
                   uint i [[thread_position_in_grid]], uint w [[threads_per_simdgroup]]) {
    int s = simd_sum(inp[i]);
    outp[2 * i] = s;
    outp[2 * i + 1] = int(w);
}
