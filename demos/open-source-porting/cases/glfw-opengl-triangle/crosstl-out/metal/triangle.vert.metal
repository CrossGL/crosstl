
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float3 vCol [[attribute(0)]];
    float2 vPos [[attribute(1)]];
};
struct VertexOutput {
    float3 color;
    float4 gl_Position [[position]];
};
// Constant Buffers
struct Uniforms {
    float4x4 MVP;
};
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]], constant Uniforms& uniforms [[buffer(0)]]) {
    VertexOutput output;
    output.gl_Position = uniforms.MVP * float4(input.vPos, 0.0, 1.0);
    output.color = input.vCol;
    return output;
}
