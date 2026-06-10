
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float4 a_position [[attribute(0)]];
    float4 a_color [[attribute(1)]];
    float2 a_texCoord0 [[attribute(2)]];
};
struct VertexOutput {
    float4 v_color;
    float2 v_texCoords;
    float4 gl_Position [[position]];
};
// Constant Buffers
struct Uniforms {
    float4x4 u_projTrans;
};
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]], constant Uniforms& uniforms [[buffer(0)]]) {
    VertexOutput output;
    output.v_color = input.a_color;
    output.v_texCoords = input.a_texCoord0;
    output.gl_Position = uniforms.u_projTrans * input.a_position;
    return output;
}
