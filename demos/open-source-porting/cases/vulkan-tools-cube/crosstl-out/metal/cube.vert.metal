
#include <metal_stdlib>
using namespace metal;

struct buf {
    float4x4 MVP;
    float4 position[12 * 3];
    float4 attr[12 * 3];
};
struct VertexOutput {
    float4 texcoord [[attribute(0)]];
    float3 frag_pos [[attribute(1)]];
    float4 gl_Position [[position]];
};
// Constant Buffers
struct Uniforms {
    buf ubuf;
};
// Vertex Shader
vertex VertexOutput vertex_main(uint _crossglVertexID [[vertex_id]], constant Uniforms& uniforms [[buffer(0)]]) {
    VertexOutput output;
    output.texcoord = uniforms.ubuf.attr[_crossglVertexID];
    output.gl_Position = uniforms.ubuf.MVP * uniforms.ubuf.position[_crossglVertexID];
    output.frag_pos = output.gl_Position.xyz;
    return output;
}
