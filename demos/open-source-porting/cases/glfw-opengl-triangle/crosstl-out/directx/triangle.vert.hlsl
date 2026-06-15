
struct VertexInput {
    float3 vCol: TEXCOORD0;
    float2 vPos: TEXCOORD1;
};
struct VertexOutput {
    float3 color: TEXCOORD0;
    float4 gl_Position: SV_POSITION;
};
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4x4 MVP;
};
// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.gl_Position = mul(MVP, float4(input.vPos, 0.0, 1.0));
    output.color = input.vCol;
    return output;
}
