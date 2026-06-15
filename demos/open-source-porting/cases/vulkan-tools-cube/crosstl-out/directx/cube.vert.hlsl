
struct buf {
    float4x4 MVP;
    float4 position[(12 * 3)];
    float4 attr[(12 * 3)];
};
struct VertexOutput {
    float4 texcoord: TEXCOORD0;
    float3 frag_pos: TEXCOORD1;
    float4 gl_Position: SV_POSITION;
};
// Constant Buffers
cbuffer Uniforms : register(b0) {
    buf ubuf;
};
// Vertex Shader
VertexOutput VSMain(uint gl_VertexIndex : SV_VertexID) {
    VertexOutput output;
    output.texcoord = ubuf.attr[gl_VertexIndex];
    output.gl_Position = mul(ubuf.MVP, ubuf.position[gl_VertexIndex]);
    output.frag_pos = output.gl_Position.xyz;
    return output;
}
