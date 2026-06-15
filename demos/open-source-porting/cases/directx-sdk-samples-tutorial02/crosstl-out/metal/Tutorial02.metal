
#include <metal_stdlib>
using namespace metal;

// Vertex Shader
struct VS_Input {
    float4 Pos [[attribute(0)]];
};

vertex float4 VS(VS_Input _crossglInput [[stage_in]]) {
    float4 Pos = _crossglInput.Pos;
    return Pos;
}

// Fragment Shader
fragment float4 PS(float4 Pos [[position]]) {
    return float4(1.0, 1.0, 0.0, 1.0);
}
