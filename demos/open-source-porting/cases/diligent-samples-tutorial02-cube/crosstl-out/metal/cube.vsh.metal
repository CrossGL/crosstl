
#include <metal_stdlib>
using namespace metal;

// Constant Buffers
struct Constants {
    float4x4 g_WorldViewProj;
};
// Vertex Shader
struct vertex_main_Input {
    float3 VSIn_Pos [[attribute(0)]];
    float4 VSIn_Color [[attribute(1)]];
};

struct vertex_main_Return {
    float4 PSIn_Pos [[position]];
    float4 PSIn_Color [[user(Color0)]];
};

vertex vertex_main_Return vertex_main(vertex_main_Input _crossglInput [[stage_in]], constant Constants& constants [[buffer(0)]]) {
    float3 VSIn_Pos = _crossglInput.VSIn_Pos;
    float4 VSIn_Color = _crossglInput.VSIn_Color;
    float4 PSIn_Pos = float4(0);
    float4 PSIn_Color = float4(0);
    PSIn_Pos = float4(VSIn_Pos, 1.0) * constants.g_WorldViewProj;
    PSIn_Color = VSIn_Color;
    vertex_main_Return _crossglOutput;
    _crossglOutput.PSIn_Pos = PSIn_Pos;
    _crossglOutput.PSIn_Color = PSIn_Color;
    return _crossglOutput;
}
