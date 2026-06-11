
#include <metal_stdlib>
using namespace metal;

struct VertexOutput {
    float4 color;
    /* CrossGL fallback: Metal vertex entry points require a position output even when the GLSL source did not write gl_Position. */
    float4 __crossgl_position [[position]];
};
// Constant Buffers
struct Material {
    int kind;
    float fa[3];
};
// Vertex Shader
vertex VertexOutput vertex_main(constant Material& material [[buffer(0)]]) {
    VertexOutput output;
    switch (material.kind) {
        case 1: {
            output.color = float4(0.2);
            break;
        }
        case 2: {
            output.color = float4(0.5);
            break;
        }
        default: {
            output.color = float4(0.0);
            break;
        }
    }
    VertexOutput _crossglReturn = output;
    _crossglReturn.__crossgl_position = float4(0.0, 0.0, 0.0, 1.0);
    return _crossglReturn;
}
