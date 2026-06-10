
struct VertexInput {
    float4 a_position: TEXCOORD0;
    float4 a_color: TEXCOORD1;
    float2 a_texCoord0: TEXCOORD2;
};
struct VertexOutput {
    float4 v_color: TEXCOORD0;
    float2 v_texCoords: TEXCOORD1;
    float4 gl_Position: SV_POSITION;
};
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4x4 u_projTrans;
};
// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.v_color = input.a_color;
    output.v_texCoords = input.a_texCoord0;
    output.gl_Position = mul(u_projTrans, input.a_position);
    return output;
}
