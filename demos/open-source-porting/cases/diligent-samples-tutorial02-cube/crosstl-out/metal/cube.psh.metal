
#include <metal_stdlib>
using namespace metal;

// Fragment Shader
struct fragment_main_Input {
    float4 PSIn_Color [[user(Color0)]];
};

struct fragment_main_Return {
    float4 PSOut_Color [[color(0)]];
};

fragment fragment_main_Return fragment_main(float4 PSIn_Pos [[position]], fragment_main_Input _crossglInput [[stage_in]]) {
    float4 PSIn_Color = _crossglInput.PSIn_Color;
    float4 PSOut_Color = float4(0);
    float4 Color = PSIn_Color;
    PSOut_Color = Color;
    fragment_main_Return _crossglOutput;
    _crossglOutput.PSOut_Color = PSOut_Color;
    return _crossglOutput;
}
