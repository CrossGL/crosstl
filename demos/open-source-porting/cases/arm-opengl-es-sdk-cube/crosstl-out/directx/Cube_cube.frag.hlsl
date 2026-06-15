
struct FragmentInput {
    float3 vv3colour: TEXCOORD0;
};
// Fragment Shader
float4 PSMain(FragmentInput input): SV_TARGET {
    float4 gl_FragColor;
    gl_FragColor = float4(input.vv3colour, 1.0);
    return gl_FragColor;
}
