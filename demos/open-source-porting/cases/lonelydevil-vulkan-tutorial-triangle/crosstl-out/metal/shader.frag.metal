
#include <metal_stdlib>
using namespace metal;

struct FragmentInput {
    float3 fragColor [[location]];
};
// Fragment Shader
fragment float4 fragment_main(FragmentInput input [[stage_in]]) {
    float4 outColor;
    outColor = float4(input.fragColor, 1.0);
    return outColor;
}
