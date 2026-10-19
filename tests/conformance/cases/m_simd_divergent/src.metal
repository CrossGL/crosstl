#include <metal_stdlib>
using namespace metal;

// simd_sum in SIMD-divergent control flow reduces over the active threads only (lanes 0..15).
kernel void hazard(device const int *inp [[buffer(0)]], device int *outp [[buffer(1)]],
                   uint i [[thread_position_in_grid]], uint w [[threads_per_simdgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    int s = -1;
    if (lane < 16u) {
        s = simd_sum(inp[i]);
    }
    outp[2 * i] = s;
    outp[2 * i + 1] = int(w);
}
