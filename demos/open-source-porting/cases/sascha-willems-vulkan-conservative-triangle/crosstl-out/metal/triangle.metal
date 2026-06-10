
#include <metal_stdlib>
using namespace metal;

struct UBO {
    float4x4 projection;
    float4x4 model;
};
struct gl_PerVertex {
    float4 gl_Position;
};
struct VertexInput {
    float3 inPos [[attribute(0)]];
    float3 inColor [[attribute(1)]];
};
struct VertexOutput {
    float3 outColor [[attribute(0)]];
    float4 gl_Position [[position]];
};
// Constant Buffers
struct Uniforms {
    UBO ubo;
};
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]], constant Uniforms& uniforms [[buffer(0)]]) {
    VertexOutput output;
    output.outColor = input.inColor;
    output.gl_Position = uniforms.ubo.projection * uniforms.ubo.model * float4(input.inPos, 1.0);
    return output;
}
