
struct FragmentInput {
    float3 fragColor: location;
};
// Fragment Shader
float4 PSMain(FragmentInput input): SV_Target0 {
    float4 outColor;
    outColor = float4(input.fragColor, 1.0);
    return outColor;
}

