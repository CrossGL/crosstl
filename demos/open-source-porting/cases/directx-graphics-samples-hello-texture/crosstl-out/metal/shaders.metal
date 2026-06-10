
#include <metal_stdlib>
using namespace metal;

struct PSInput {
    float4 position [[position]];
    float2 uv [[attribute(4)]];
};
// Vertex Shader
struct VSMain_Input {
    float4 position [[attribute(0)]];
    float2 uv [[attribute(4)]];
};

vertex PSInput VSMain(VSMain_Input _crossglInput [[stage_in]], texture2d<float> g_texture [[texture(0)]]) {
    float4 position = _crossglInput.position;
    float2 uv = _crossglInput.uv;
    PSInput result;
    result.position = position;
    result.uv = uv;
    return result;
}

// Fragment Shader
fragment float4 PSMain(PSInput input [[stage_in]], texture2d<float> g_texture [[texture(0)]], sampler g_sampler [[sampler(0)]]) {
    return g_texture.sample(g_sampler, input.uv);
}

