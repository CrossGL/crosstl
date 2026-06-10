
struct FragmentInput {
    float2 v_texCoord: TEXCOORD0;
};
Texture2D s_texture : register(t0);
SamplerState s_textureSampler : register(s0);
// Fragment Shader
float4 PSMain(FragmentInput input): SV_TARGET {
    float4 gl_FragColor;
    gl_FragColor = s_texture.Sample(s_textureSampler, input.v_texCoord);
    return gl_FragColor;
}
