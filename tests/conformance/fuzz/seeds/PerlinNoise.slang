struct VertexInput
{
    float3 position : POSITION;
};

struct VertexOutput
{
    float2 uv : TEXCOORD0;
    float4 position : SV_Position;
};

struct FragmentInput
{
    float2 uv : TEXCOORD0;
};

struct FragmentOutput
{
    float4 color : SV_Target;
};

// Vertex Shader
[shader("vertex")] VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.uv = input.position.xy * 10.0;
    output.position = float4(input.position, 1.0);
    return output;
}

// Fragment Shader
float perlinNoise(float2 p)
{
    return frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
}

[shader("fragment")] FragmentOutput PSMain(FragmentInput input) {
    FragmentOutput output;
    float noise = perlinNoise(input.uv);
    float height = noise * 10.0;
    float3 color = float3(height / 10.0, 1.0 - height / 10.0, 0.0);
    output.color = float4(color, 1.0);
    return output;
}
