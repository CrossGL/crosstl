
#include <metal_stdlib>
using namespace metal;

struct PSInput {
    float4 position [[position]];
    float4 color [[attribute(0)]];
};
// Vertex Shader
struct VSMain_Input {
    float4 position [[attribute(0)]];
    float4 color [[attribute(1)]];
};

vertex PSInput VSMain(VSMain_Input _crossglInput [[stage_in]]) {
    float4 position = _crossglInput.position;
    float4 color = _crossglInput.color;
    PSInput result;
    result.position = position;
    result.color = color;
    return result;
}

// Fragment Shader
fragment float4 PSMain(PSInput input [[stage_in]]) {
    return input.color;
}
