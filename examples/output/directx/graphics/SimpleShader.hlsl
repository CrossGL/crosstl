
struct VertexInput
{
    float3 position;
    float2 texCoord;
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
VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.uv = input.texCoord;
    output.position = float4(input.position, 1.0);
    return output;
}

// Fragment Shader
FragmentOutput main(FragmentInput input)
{
    FragmentOutput output;
    float r = input.uv.x;
    float g = input.uv.y;
    float b = 0.5;
    output.color = float4(r, g, b, 1.0);
    return output;
}
