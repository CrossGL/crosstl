
#include <metal_stdlib>
using namespace metal;

struct VertexOutput {
    float4 color;
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
    return output;
}
