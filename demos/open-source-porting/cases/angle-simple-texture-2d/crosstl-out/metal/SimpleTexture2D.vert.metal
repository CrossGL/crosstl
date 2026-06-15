
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float4 a_position [[attribute(0)]];
    float2 a_texCoord [[attribute(1)]];
};
struct VertexOutput {
    float2 v_texCoord;
    float4 gl_Position [[position]];
};
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]]) {
    VertexOutput output;
    output.gl_Position = input.a_position;
    output.v_texCoord = input.a_texCoord;
    return output;
}
