#include <metal_stdlib>
using namespace metal;
// Hazard H15: half fma rounds the exact a * b + c once (IEEE 754-2019 fusedMultiplyAdd).
kernel void hazard(device const float* a [[buffer(0)]], device float* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    half x = half(a[3 * i]);
    half y = half(a[3 * i + 1]);
    half z = half(a[3 * i + 2]);
    o[i] = float(fma(x, y, z));
}
