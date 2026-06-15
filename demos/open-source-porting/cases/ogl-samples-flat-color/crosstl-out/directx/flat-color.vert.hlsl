
struct VertexInput {
    float4 Position: TEXCOORD0;
};
struct VertexOutput {
    float4 gl_Position: SV_POSITION;
};
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4x4 MVP;
};
// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.gl_Position = mul(MVP, input.Position);
    return output;
}
