
struct PSInput {
    float4 position: SV_POSITION;
    float2 uv: TexCoord;
};
Texture2D g_texture : register(t0);
SamplerState g_sampler : register(s0);
// Vertex Shader
PSInput VSMain(float4 position : Position, float2 uv : TexCoord) {
    PSInput result;
    result.position = position;
    result.uv = uv;
    return result;
}

// Fragment Shader
float4 PSMain(PSInput input): SV_TARGET {
    return g_texture.Sample(g_sampler, input.uv);
}
