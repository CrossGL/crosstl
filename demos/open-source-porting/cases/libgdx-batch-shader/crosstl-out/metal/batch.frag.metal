
#include <metal_stdlib>
using namespace metal;

struct FragmentInput {
    float4 v_color;
    float2 v_texCoords;
};
// Fragment Shader
fragment float4 fragment_main(FragmentInput input [[stage_in]], texture2d<float> u_texture [[texture(0)]]) {
    float4 gl_FragColor;
    gl_FragColor = input.v_color * u_texture.sample(sampler(mag_filter::linear, min_filter::linear), input.v_texCoords);
    return gl_FragColor;
}

