
struct VSOutput {
    float4 position: SV_POSITION;
    float4 color: Color0;
    float2 texCoord: TexCoord0;
};
Texture2D Texture : register(t0);
SamplerState TextureSampler : register(s0);
float4x4 MatrixTransform;
VSOutput SpriteVertexShader(float4 position : Position, float4 color : Color0, float2 texCoord : TexCoord0) {
    VSOutput output;
    output.position = mul(position, MatrixTransform);
    output.color = color;
    output.texCoord = texCoord;
    return output;
}

float4 SpritePixelShader(VSOutput input): SV_TARGET0 {
    return (Texture.Sample(TextureSampler, input.texCoord) * input.color);
}

// Fragment Shader
void PSMain() {
}

