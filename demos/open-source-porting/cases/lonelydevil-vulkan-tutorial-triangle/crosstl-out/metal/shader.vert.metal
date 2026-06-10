
#include <metal_stdlib>
using namespace metal;

struct VertexOutput {
    float3 fragColor [[location]];
    float4 gl_Position [[position]];
};
constant float2 positions[3] = {float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5)};
constant float3 colors[3] = {float3(0.5, 0.5, 0.0), float3(0.0, 0.5, 0.5), float3(0.5, 0.0, 0.5)};
// Vertex Shader
vertex VertexOutput vertex_main(uint _crossglVertexID [[vertex_id]]) {
    VertexOutput output;
    output.gl_Position = float4(positions[_crossglVertexID], 0.0, 1.0);
    output.fragColor = colors[_crossglVertexID];
    return output;
}

