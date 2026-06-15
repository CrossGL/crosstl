
struct VertexInput {
    float4 a_position: TEXCOORD0;
    float2 a_texCoord: TEXCOORD1;
};
struct VertexOutput {
    float2 v_texCoord: TEXCOORD0;
    float4 gl_Position: SV_POSITION;
};
// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.gl_Position = input.a_position;
    output.v_texCoord = input.a_texCoord;
    return output;
}
