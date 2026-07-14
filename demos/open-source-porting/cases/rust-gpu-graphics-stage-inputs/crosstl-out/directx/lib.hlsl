
float3 tonemap(float3 color);

float3 tonemap(float3 color) {
    return color;
}

// Fragment Shader
void PSMain(out float4 output : SV_TARGET) {
    float3 color = float3(1.0, 0.5, 0.25);
    output = float4(tonemap(color), 1.0);
}

// Vertex Shader
void VSMain(float2 pos : TEXCOORD0, out float4 builtin_pos : SV_POSITION) {
    builtin_pos = float4(float3(pos, 0.0), 1.0);
}
