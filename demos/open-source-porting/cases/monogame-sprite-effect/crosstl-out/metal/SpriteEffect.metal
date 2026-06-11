
#include <metal_stdlib>
using namespace metal;

struct VSOutput {
    float4 position [[position]];
    float4 color [[user(Color0)]];
    float2 texCoord [[attribute(5)]];
};
// Constant Buffers
struct HlslProgramConstants {
    float4x4 MatrixTransform;
};
VSOutput SpriteVertexShader(float4 position [[attribute(0)]], float4 color [[user(Color0)]], float2 texCoord [[attribute(5)]], constant HlslProgramConstants& hlslProgramConstants) {
    VSOutput output;
    output.position = position * hlslProgramConstants.MatrixTransform;
    output.color = color;
    output.texCoord = texCoord;
    return output;
}

float4 SpritePixelShader(VSOutput input, texture2d<float> Texture) {
    return Texture.sample(sampler(mag_filter::linear, min_filter::linear), input.texCoord) * input.color;
}

// Fragment Shader
fragment void fragment_main(constant HlslProgramConstants& hlslProgramConstants [[buffer(0)]], texture2d<float> Texture [[texture(0)]]) {
}
