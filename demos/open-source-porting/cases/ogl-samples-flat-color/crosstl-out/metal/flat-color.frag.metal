
#include <metal_stdlib>
using namespace metal;

// Constant Buffers
struct Uniforms {
    float4 Diffuse;
};
// Fragment Shader
fragment float4 fragment_main(constant Uniforms& uniforms [[buffer(0)]]) {
    float4 Color;
    Color = uniforms.Diffuse;
    return Color;
}
