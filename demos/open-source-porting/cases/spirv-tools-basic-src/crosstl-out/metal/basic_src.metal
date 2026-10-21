
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float4 ua_position [[attribute(0)]];
};
struct VertexOutput {
    float4 ANGLEXfbPosition [[user(locn0)]];
    float4 position [[position]];
};
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]]) {
    VertexOutput output;
    output.position = input.ua_position;
    return output;
}
