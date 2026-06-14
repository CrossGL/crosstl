
#include <metal_stdlib>
using namespace metal;

struct FragmentInput {
    float3 color;
};
// Fragment Shader
fragment float4 fragment_main(FragmentInput input [[stage_in]]) {
    float4 gl_FragColor;
    gl_FragColor = float4(input.color, 1.0);
    return gl_FragColor;
}
