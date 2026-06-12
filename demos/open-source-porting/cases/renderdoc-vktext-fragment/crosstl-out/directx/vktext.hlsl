
struct FragmentInput {
    float4 tex: TEXCOORD0;
    float2 glyphuv: TEXCOORD1;
};
Texture2D tex0 : register(t3);
SamplerState tex0Sampler : register(s0);
// Fragment Shader
float4 PSMain(FragmentInput input): SV_Target0 {
    float4 color_out;
    float text = 0.0;
    if (((((input.glyphuv.x >= 0.0) && (input.glyphuv.x <= 1.0)) && (input.glyphuv.y >= 0.0)) && (input.glyphuv.y <= 1.0))) {
        float2 uv;
        uv.x = lerp(input.tex.x, input.tex.z, input.glyphuv.x);
        uv.y = lerp(input.tex.y, input.tex.w, input.glyphuv.y);
        text = tex0.Sample(tex0Sampler, uv.xy).x;
    }
    color_out = float4(float3(text, text, text), clamp((text + 0.5), 0.0, 1.0));
    return color_out;
}
