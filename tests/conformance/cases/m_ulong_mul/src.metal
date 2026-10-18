#include <metal_stdlib>
using namespace metal;
// Hazard H15: 64-bit unsigned product; the high word is observable.
kernel void hazard(device const uint* a [[buffer(0)]], device uint* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    ulong v = ulong(a[i]) * 4294967295ul;
    o[i] = uint(v >> 32);
}
