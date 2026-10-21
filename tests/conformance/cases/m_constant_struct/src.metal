#include <metal_stdlib>
using namespace metal;
// MSL constant layout: float3 takes 16 bytes, so bias is at 16; the packed_float3 shift follows at 20; the float2
// array has an 8-byte stride from 32; float3x3 columns are 16 bytes apart from 64; mode is at 112, the float array
// at 116 and 120, and tail at 128.
struct Params {
    float3 scale;
    float bias;
    packed_float3 shift;
    float2 offsets[3];
    float3x3 rot;
    int mode;
    float weights[2];
    float2 tail;
};
kernel void hazard(device const float* x [[buffer(0)]], device float* o [[buffer(1)]],
                   constant Params& params [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    float v = x[i];
    float3 r = params.rot * float3(v, 1.0, 2.0);
    o[8 * i + 0] = v * params.scale.x + params.bias;
    o[8 * i + 1] = v * params.scale.y + params.scale.z + params.shift.y;
    o[8 * i + 2] = params.offsets[i % 3].x;
    o[8 * i + 3] = params.offsets[i % 3].y + params.shift.z;
    o[8 * i + 4] = r[i % 3];
    o[8 * i + 5] = float(params.mode) + params.shift.x;
    o[8 * i + 6] = params.weights[i % 2];
    o[8 * i + 7] = params.tail.y;
}
