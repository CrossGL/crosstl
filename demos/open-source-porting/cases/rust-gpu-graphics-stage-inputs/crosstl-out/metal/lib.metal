
#include <metal_stdlib>
using namespace metal;

float3 tonemap(float3 color)  {
    return color;
}

// Fragment Shader
struct fragment_main_fs_Return {
    float4 output [[color(0)]];
};

fragment fragment_main_fs_Return fragment_main_fs() {
    float4 output = float4(0);
    float3 color = float3(1.0, 0.5, 0.25);
    output = float4(tonemap(color), 1.0);
    fragment_main_fs_Return _crossglOutput;
    _crossglOutput.output = output;
    return _crossglOutput;
}

// Vertex Shader
struct vertex_main_vs_Input {
    float2 pos [[attribute(0)]];
};

struct vertex_main_vs_Return {
    float4 builtin_pos [[position]];
};

vertex vertex_main_vs_Return vertex_main_vs(vertex_main_vs_Input _crossglInput [[stage_in]]) {
    float2 pos = _crossglInput.pos;
    float4 builtin_pos = float4(0);
    builtin_pos = float4(float3(pos, 0.0), 1.0);
    vertex_main_vs_Return _crossglOutput;
    _crossglOutput.builtin_pos = builtin_pos;
    return _crossglOutput;
}
