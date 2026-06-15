
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4 Diffuse;
};
// Fragment Shader
float4 PSMain(): SV_Target0 {
    float4 Color;
    Color = Diffuse;
    return Color;
}
