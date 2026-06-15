
static const float3 lightDir = float3(0.424, 0.566, 0.707);

struct FragmentInput {
    float4 texcoord: TEXCOORD0;
    float3 frag_pos: TEXCOORD1;
};
Texture2D tex : register(t1);
SamplerState texSampler : register(s0);
float linearToSrgb(float linear_) {
    if ((linear_ <= 0.0031308)) {
        return (linear_ * 12.92);
    } else {
        return ((1.055 * pow(linear_, (1.0 / 2.4))) - 0.055);
    }
    return float(0);
}

float3 linearToSrgb(float3 linear_) {
    return float3(linearToSrgb(linear_.r), linearToSrgb(linear_.g), linearToSrgb(linear_.b));
}

float4 linearToSrgb(float4 linear_) {
    return float4(linearToSrgb(linear_.rgb), linear_.a);
}

// Fragment Shader
float4 PSMain(FragmentInput input): SV_Target0 {
    float4 uFragColor;
    float3 dX = ddx(input.frag_pos);
    float3 dY = ddy(input.frag_pos);
    float3 normal = normalize(cross(dX, dY));
    float light = max(0.0, dot(lightDir, normal));
    uFragColor = linearToSrgb((light * tex.Sample(texSampler, input.texcoord.xy)));
    return uFragColor;
}
