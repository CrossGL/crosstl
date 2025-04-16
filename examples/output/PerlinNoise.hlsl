
struct VertexInput
{
    float3 position;
};
struct VertexOutput
{
    float2 uv;
    float4 position;
};
struct FragmentInput
{
    float2 uv;
};
struct FragmentOutput
{
    float4 color;
};
// Vertex Shader
// Vertex Shader
VertexOutput VSMain(VertexInput input)
{
    output;
    output.uv = input.position.xy * 10.0;
    output.position = float4(input.position, 1.0);
    return output;
}

// Fragment Shader
// Fragment Shader
FragmentOutput PSMain(FragmentInput input)
{
    output;
    float noise = perlinNoise(input.uv);
    float3 color = float3(height / 10.0, 1.0 - height / 10.0, 0.0);
    return output;
}
