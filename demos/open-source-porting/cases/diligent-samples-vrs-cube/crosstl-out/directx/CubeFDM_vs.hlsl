
struct g_Constants {
    float4x4 WorldViewProj;
    int PrimitiveShadingRate;
    int DrawMode;
};
struct VertexInput {
    float3 in_Pos: TEXCOORD0;
    float2 in_UV: TEXCOORD1;
};
struct VertexOutput {
    float2 out_UV: TEXCOORD0;
    float4 gl_Position: SV_POSITION;
};
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4x4 WorldViewProj;
    int PrimitiveShadingRate;
    int DrawMode;
};
// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.gl_Position = mul(float4(input.in_Pos, 1.0), WorldViewProj);
    output.out_UV = input.in_UV;
    return output;
}
