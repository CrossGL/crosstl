
#include <metal_stdlib>
using namespace metal;

struct FragmentInput {
    float4 vA [[user(locn0)]];
    float vB [[user(locn1)]];
};
// Fragment Shader
fragment float4 fragment_main(FragmentInput input [[stage_in]]) {
    float4 FragColor;
    FragColor = round(input.vA);
    FragColor *= round(input.vB);
    return FragColor;
}
