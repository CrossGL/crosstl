
#include <metal_stdlib>
using namespace metal;

struct VertexOutput {
    float4 texcoord [[user(locn0)]];
    float3 frag_pos [[user(locn1)]];
    float4 gl_Position [[position]];
};
// Constant Buffers
struct buf {
    float4x4 MVP;
    float4 position[36];
    float4 attr[36];
};
// Vertex Shader
vertex VertexOutput vertex_main(uint _crossglVertexID [[vertex_id]], constant buf& buf [[buffer(0)]]) {
    VertexOutput output;
    output.texcoord = buf.attr[_crossglVertexID];
    output.gl_Position = buf.MVP * buf.position[_crossglVertexID];
    output.frag_pos = output.gl_Position.xyz;
    return output;
}
