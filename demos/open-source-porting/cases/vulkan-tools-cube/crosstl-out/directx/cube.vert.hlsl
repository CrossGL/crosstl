
struct VertexOutput {
    float4 texcoord: TEXCOORD0;
    float3 frag_pos: TEXCOORD1;
    float4 gl_Position: SV_POSITION;
};
// Constant Buffers
cbuffer buf : register(b0) {
    float4x4 MVP;
    float4 position[36];
    float4 attr[36];
};
// Vertex Shader
VertexOutput VSMain(uint gl_VertexIndex : SV_VertexID) {
    VertexOutput output;
    output.texcoord = attr[gl_VertexIndex];
    output.gl_Position = mul(MVP, position[gl_VertexIndex]);
    output.frag_pos = output.gl_Position.xyz;
    return output;
}
