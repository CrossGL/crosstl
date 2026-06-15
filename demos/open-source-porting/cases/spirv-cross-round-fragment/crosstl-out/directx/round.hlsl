
struct FragmentInput {
    float4 vA: TEXCOORD0;
    float vB: TEXCOORD1;
};
// Fragment Shader
float4 PSMain(FragmentInput input): SV_Target0 {
    float4 FragColor;
    FragColor = round(input.vA);
    FragColor *= round(input.vB);
    return FragColor;
}
