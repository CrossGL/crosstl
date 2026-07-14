
struct FragmentInput {
    float3 nearPoint: TEXCOORD0;
    float3 farPoint: TEXCOORD1;
};
float4 grid(float3 pos);

float4 grid(float3 pos) {
    float2 coord = pos.xz;
    float2 derivative = fwidth(coord);
    float2 gridLine = (abs((frac((coord - 0.5)) - 0.5)) / derivative);
    float line_ = min(gridLine.x, gridLine.y);
    return float4(0.5, 0.5, 0.5, (1.0 - min(line_, 1.0)));
}

// Fragment Shader
float4 PSMain(FragmentInput input): SV_Target0 {
    float4 outColor;
    float t = (-input.nearPoint.y / (input.farPoint.y - input.nearPoint.y));
    float3 pos = (input.nearPoint + (t * (input.farPoint - input.nearPoint)));
    outColor = grid(pos);
    return outColor;
}
