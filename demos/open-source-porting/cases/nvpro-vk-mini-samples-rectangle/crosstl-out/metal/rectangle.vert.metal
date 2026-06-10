
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float2 inPosition [[attribute(0)]];
    float3 inColor [[attribute(1)]];
};
struct VertexOutput {
    float3 outFragColor [[attribute(0)]];
    float4 gl_Position [[position]];
};
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]]) {
    VertexOutput output;
    output.gl_Position = float4(input.inPosition, 0.0, 1.0);
    output.outFragColor = input.inColor;
    return output;
}

