
#include <metal_stdlib>
using namespace metal;

struct FragmentInput {
    float4 vA [[attribute(0)]];
    float vB [[attribute(1)]];
};
// Fragment Shader
fragment float4 fragment_main(FragmentInput input [[stage_in]]) {
    float4 FragColor;
    FragColor = round(input.vA);
    FragColor *= round(input.vB);
    return FragColor;
}
