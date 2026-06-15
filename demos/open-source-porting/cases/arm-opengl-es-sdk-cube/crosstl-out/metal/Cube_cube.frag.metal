
#include <metal_stdlib>
using namespace metal;

struct FragmentInput {
    float3 vv3colour;
};
// Fragment Shader
fragment float4 fragment_main(FragmentInput input [[stage_in]]) {
    float4 gl_FragColor;
    gl_FragColor = float4(input.vv3colour, 1.0);
    return gl_FragColor;
}
