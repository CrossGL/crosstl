
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float4 av4position [[attribute(0)]];
    float3 av3colour [[attribute(1)]];
};
struct VertexOutput {
    float3 vv3colour;
    float4 gl_Position [[position]];
};
// Constant Buffers
struct Uniforms {
    float4x4 mvp;
};
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]], constant Uniforms& uniforms [[buffer(0)]]) {
    VertexOutput output;
    output.vv3colour = input.av3colour;
    output.gl_Position = uniforms.mvp * input.av4position;
    return output;
}
