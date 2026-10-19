#include <metal_stdlib>
using namespace metal;
// Hazard H4: int < uint converts the int to uint (C++ usual arithmetic conversions); -1 < 4u is false.
kernel void hazard(device const int* a [[buffer(0)]], device int* o [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    int x = a[i];
    uint n = 4u;
    o[i] = (x < n) ? 1 : 0;
}
