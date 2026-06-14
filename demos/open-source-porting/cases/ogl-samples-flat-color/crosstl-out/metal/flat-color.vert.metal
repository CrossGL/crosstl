
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float4 Position [[attribute(0)]];
};
struct VertexOutput {
    float4 gl_Position [[position]];
};
// Constant Buffers
struct Uniforms {
    float4x4 MVP;
};
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]], constant Uniforms& uniforms [[buffer(0)]]) {
    VertexOutput output;
    output.gl_Position = uniforms.MVP * input.Position;
    return output;
}
