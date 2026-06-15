
#include <metal_stdlib>
using namespace metal;

struct FragmentInput {
    float4 tex [[attribute(0)]];
    float2 glyphuv [[attribute(1)]];
};
// Fragment Shader
fragment float4 fragment_main(FragmentInput input [[stage_in]], texture2d<float> tex0 [[texture(3)]]) {
    float4 color_out;
    float text = 0.0;
    if (input.glyphuv.x >= 0.0 && input.glyphuv.x <= 1.0 && input.glyphuv.y >= 0.0 && input.glyphuv.y <= 1.0) {
        float2 uv;
        uv.x = mix(input.tex.x, input.tex.z, input.glyphuv.x);
        uv.y = mix(input.tex.y, input.tex.w, input.glyphuv.y);
        text = tex0.sample(sampler(mag_filter::linear, min_filter::linear), uv.xy).x;
    }
    color_out = float4(float3(text), clamp(text + 0.5, 0.0, 1.0));
    return color_out;
}
