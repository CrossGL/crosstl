
#include <metal_stdlib>
using namespace metal;

struct FragmentInput {
    float2 fragTexCoord;
    float4 fragColor;
};
// Constant Buffers
struct Uniforms {
    float4 colDiffuse;
};
// Fragment Shader
fragment float4 fragment_main(FragmentInput input [[stage_in]], constant Uniforms& uniforms [[buffer(0)]], texture2d<float> texture0 [[texture(0)]]) {
    float4 finalColor;
    float4 texelColor = texture0.sample(sampler(mag_filter::linear, min_filter::linear), input.fragTexCoord);
    finalColor = texelColor * uniforms.colDiffuse * input.fragColor;
    return finalColor;
}
