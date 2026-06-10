
#include <metal_stdlib>
using namespace metal;

// Fragment Shader
struct fragment_main_fs_Return {
    float4 out_frag_color [[color(0)]];
};

fragment fragment_main_fs_Return fragment_main_fs() {
    float4 out_frag_color = float4(0);
    out_frag_color = float4(1.0, 1.0, 1.0, 1.0);
    fragment_main_fs_Return _crossglOutput;
    _crossglOutput.out_frag_color = out_frag_color;
    return _crossglOutput;
}
