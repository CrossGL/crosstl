
#include <metal_stdlib>
using namespace metal;

__attribute__((unused)) constant float3 lightDir = float3(0.424, 0.566, 0.707);

struct FragmentInput {
    float4 texcoord [[attribute(0)]];
    float3 frag_pos [[attribute(1)]];
};
float linearToSrgb(float linear)  {
    if (linear <= 0.0031308) {
        return linear * 12.92;
    } else {
        return 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
    }
    return float(0) /* fallback for unmatched generated control flow */;
}

float3 linearToSrgb(float3 linear)  {
    return float3(linearToSrgb(linear.r), linearToSrgb(linear.g), linearToSrgb(linear.b));
}

float4 linearToSrgb(float4 linear)  {
    return float4(linearToSrgb(linear.rgb), linear.a);
}

// Fragment Shader
fragment float4 fragment_main(FragmentInput input [[stage_in]], texture2d<float> tex [[texture(1)]]) {
    float4 uFragColor;
    float3 dX = dfdx(input.frag_pos);
    float3 dY = dfdy(input.frag_pos);
    float3 normal = normalize(cross(dX, dY));
    float light = max(0.0, dot(lightDir, normal));
    uFragColor = linearToSrgb(float4(light) * tex.sample(sampler(mag_filter::linear, min_filter::linear), input.texcoord.xy));
    return uFragColor;
}

