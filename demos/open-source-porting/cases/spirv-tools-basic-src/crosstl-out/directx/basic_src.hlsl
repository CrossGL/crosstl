
struct VertexInput {
    float4 ua_position: TEXCOORD0;
};
struct VertexOutput {
    float4 position: SV_POSITION;
};
// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.position = input.ua_position;
    return output;
}
