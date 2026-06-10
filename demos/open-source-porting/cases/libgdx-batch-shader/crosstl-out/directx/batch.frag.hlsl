
struct FragmentInput {
    float4 v_color: TEXCOORD0;
    float2 v_texCoords: TEXCOORD1;
};
Texture2D u_texture : register(t0);
SamplerState u_textureSampler : register(s0);
// Fragment Shader
float4 PSMain(FragmentInput input): SV_TARGET {
    float4 gl_FragColor;
    gl_FragColor = (input.v_color * u_texture.Sample(u_textureSampler, input.v_texCoords));
    return gl_FragColor;
}

