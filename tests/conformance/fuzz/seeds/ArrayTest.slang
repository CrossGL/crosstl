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

[[vk::binding(0, 0)]] cbuffer TestBuffer : register(b0)
{
    float values[4];
    float3 colors[2];
};

// Vertex Shader
[shader("vertex")] VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.uv = input.texCoord;
    float scale = values[0] + values[1];
    float3 position = input.position * scale;
    output.position = float4(position, 1.0);
    return output;
}

    // Fragment Shader
    [shader("fragment")] FragmentOutput PSMain(FragmentInput input)
{
    FragmentOutput output;
    float3 color = colors[0];
    if (input.uv.x > 0.5)
    {
        color = colors[1];
    }
    output.color = float4(color, 1.0);
    return output;
}
