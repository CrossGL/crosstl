
struct VertexInput {
    float3 vertexPosition: POSITION;
    float2 vertexTexCoord: TEXCOORD0;
    float3 vertexNormal: TEXCOORD1;
    float4 vertexColor: TEXCOORD2;
};
struct VertexOutput {
    float2 fragTexCoord: TEXCOORD0;
    float4 fragColor: TEXCOORD1;
    float4 gl_Position: SV_POSITION;
};
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4x4 mvp;
};
// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.fragTexCoord = input.vertexTexCoord;
    output.fragColor = input.vertexColor;
    output.gl_Position = mul(mvp, float4(input.vertexPosition, 1.0));
    return output;
}
