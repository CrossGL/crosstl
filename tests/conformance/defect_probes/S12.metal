#include <metal_stdlib>
using namespace metal;
kernel void k(device const ulong* a [[buffer(0)]], device ulong* o [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  o[i] = a[i] * 3ul; }
