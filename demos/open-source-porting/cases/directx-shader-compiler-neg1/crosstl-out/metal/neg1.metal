
#include <metal_stdlib>
using namespace metal;

// Fragment Shader
struct fragment_main_Input {
    float4 a [[attribute(0)]];
};

fragment float4 fragment_main(fragment_main_Input _crossglInput [[stage_in]]) {
    float4 a = _crossglInput.a;
    return -a.yxxx;
}
