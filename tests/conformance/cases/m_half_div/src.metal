#include <metal_stdlib>
using namespace metal;
// Hazard H15: half division is correctly rounded to binary16 (MSL section 8.2, Table 8.1 in safe mode).
kernel void hazard(device const float* a [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    half x = half(a[2 * i]);
    half y = half(a[2 * i + 1]);
    o[i] = float(x / y);
}
