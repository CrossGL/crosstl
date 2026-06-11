
struct FragmentInput {
    float2 fragTexCoord: TEXCOORD0;
    float4 fragColor: TEXCOORD1;
};
Texture2D texture0 : register(t0);
SamplerState texture0Sampler : register(s0);
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4 colDiffuse;
};
// Fragment Shader
float4 PSMain(FragmentInput input): SV_Target0 {
    float4 finalColor;
    float4 texelColor = texture0.Sample(texture0Sampler, input.fragTexCoord);
    finalColor = ((texelColor * colDiffuse) * input.fragColor);
    return finalColor;
}
