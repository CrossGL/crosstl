struct VertexInput
{
    float3 position : POSITION;
    float2 texCoord : TEXCOORD0;
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
    output.uv = input.texCoord;
    output.position = float4(input.position, 1.0);
    return output;
}

    // Fragment Shader
    [shader("fragment")] FragmentOutput PSMain(FragmentInput input)
{
    FragmentOutput output;
    float r = input.uv.x;
    float g = input.uv.y;
    float b = 0.5;
    output.color = float4(r, g, b, 1.0);
    return output;
}
