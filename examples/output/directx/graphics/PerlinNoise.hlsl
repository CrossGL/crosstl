
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
VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.uv = (input.position.xy * 10.0);
    output.position = float4(input.position, 1.0);
    return output;
}

// Fragment Shader
FragmentOutput main(FragmentInput input)
{
    FragmentOutput output;
    float noise = perlinNoise(input.uv);
    float height = (noise * 10.0);
    float3 color = float3((height / 10.0), (1.0 - (height / 10.0)), 0.0);
    output.color = float4(color, 1.0);
    return output;
}

float perlinNoise(VectorType(element_type = PrimitiveType(name = float, size_bits = None), size = 2) p)
{
    return fract((sin(dot(p, float2(12.9898, 78.233))) * 43758.5453));
}
