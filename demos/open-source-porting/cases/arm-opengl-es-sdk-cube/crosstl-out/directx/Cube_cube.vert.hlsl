
struct VertexInput {
    float4 av4position: TEXCOORD0;
    float3 av3colour: TEXCOORD1;
};
struct VertexOutput {
    float3 vv3colour: TEXCOORD0;
    float4 gl_Position: SV_POSITION;
};
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4x4 mvp;
};
// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.vv3colour = input.av3colour;
    output.gl_Position = mul(mvp, input.av4position);
    return output;
}
