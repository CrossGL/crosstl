
struct FragmentInput {
    float3 color: TEXCOORD0;
};
// Fragment Shader
float4 PSMain(FragmentInput input): SV_TARGET {
    float4 gl_FragColor;
    gl_FragColor = float4(input.color, 1.0);
    return gl_FragColor;
}
