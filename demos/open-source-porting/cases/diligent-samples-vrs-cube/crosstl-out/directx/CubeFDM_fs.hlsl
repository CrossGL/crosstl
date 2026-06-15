
struct g_Constants {
    float4x4 WorldViewProj;
    int PrimitiveShadingRate;
    int DrawMode;
};
struct FragmentInput {
    float2 in_UV: TEXCOORD0;
};
Texture2D g_Texture : register(t0);
SamplerState g_TextureSampler : register(s0);
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4x4 WorldViewProj;
    int PrimitiveShadingRate;
    int DrawMode;
};
uint2 CrossGLFragmentSizeFromShadingRate(uint shadingRate) {
    uint width = 1u << ((shadingRate >> 2u) & 3u);
    uint height = 1u << (shadingRate & 3u);
    return uint2(width, height);
}

float4 FragmentDensityToColor(uint2 _crossglFragSize) {
    float h = (clamp((1.0 - (1.0 / float((_crossglFragSize.x * _crossglFragSize.y)))), 0.0, 1.0) / 1.35);
    float3 col = float3((abs(((h * 6.0) - 3.0)) - 1.0), (2.0 - abs(((h * 6.0) - 2.0))), (2.0 - abs(((h * 6.0) - 4.0))));
    return float4(clamp(col, float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0)), 1.0);
}

// Fragment Shader
float4 PSMain(FragmentInput input, uint _crossglShadingRate : SV_ShadingRate): SV_Target0 {
    uint2 _crossglFragSize = CrossGLFragmentSizeFromShadingRate(_crossglShadingRate);
    float4 out_Color;
    float4 Col = g_Texture.Sample(g_TextureSampler, input.in_UV);
    switch (DrawMode) {
        case 0: {
            out_Color = Col;
            break;
        }
        case 1: {
            out_Color = ((Col + FragmentDensityToColor(_crossglFragSize)) * 0.5);
            break;
        }
    }
    return out_Color;
}
