
#include <metal_stdlib>
using namespace metal;

struct VSOutput {
    float4 position [[position]];
    float4 color [[attribute(0)]];
    float2 texCoord [[attribute(5)]];
};
// Constant Buffers
struct HlslProgramConstants {
    float4x4 MatrixTransform;
};
// Vertex Shader
struct SpriteVertexShader_Input {
    float4 position [[attribute(0)]];
    float4 color [[attribute(1)]];
    float2 texCoord [[attribute(5)]];
};

vertex VSOutput SpriteVertexShader(SpriteVertexShader_Input _crossglInput [[stage_in]], constant HlslProgramConstants& hlslProgramConstants [[buffer(0)]], texture2d<float> Texture [[texture(0)]]) {
    float4 position = _crossglInput.position;
    float4 color = _crossglInput.color;
    float2 texCoord = _crossglInput.texCoord;
    VSOutput output;
    output.position = position * hlslProgramConstants.MatrixTransform;
    output.color = color;
    output.texCoord = texCoord;
    return output;
}

// Fragment Shader
fragment float4 SpritePixelShader(VSOutput input [[stage_in]], constant HlslProgramConstants& hlslProgramConstants [[buffer(0)]], texture2d<float> Texture [[texture(0)]]) {
    return Texture.sample(sampler(mag_filter::linear, min_filter::linear), input.texCoord) * input.color;
}
