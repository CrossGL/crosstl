
#include <metal_stdlib>
using namespace metal;

struct FragmentInput {
    float2 v_texCoord;
};
// Fragment Shader
fragment float4 fragment_main(FragmentInput input [[stage_in]], texture2d<float> s_texture [[texture(0)]]) {
    float4 gl_FragColor;
    gl_FragColor = s_texture.sample(sampler(mag_filter::linear, min_filter::linear), input.v_texCoord);
    return gl_FragColor;
}
