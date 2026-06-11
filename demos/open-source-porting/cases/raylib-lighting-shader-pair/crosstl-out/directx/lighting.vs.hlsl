
struct VertexInput {
    float3 vertexPosition: POSITION;
    float2 vertexTexCoord: TEXCOORD0;
    float3 vertexNormal: TEXCOORD1;
    float4 vertexColor: TEXCOORD2;
};
struct VertexOutput {
    float3 fragPosition: TEXCOORD0;
    float2 fragTexCoord: TEXCOORD1;
    float4 fragColor: TEXCOORD2;
    float3 fragNormal: TEXCOORD3;
    float4 gl_Position: SV_POSITION;
};
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4x4 mvp;
    float4x4 matModel;
    float4x4 matNormal;
};
// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.fragPosition = float3(mul(matModel, float4(input.vertexPosition, 1.0)).xyz);
    output.fragTexCoord = input.vertexTexCoord;
    output.fragColor = input.vertexColor;
    output.fragNormal = normalize(float3(mul(matNormal, float4(input.vertexNormal, 1.0)).xyz));
    output.gl_Position = mul(mvp, float4(input.vertexPosition, 1.0));
    return output;
}
