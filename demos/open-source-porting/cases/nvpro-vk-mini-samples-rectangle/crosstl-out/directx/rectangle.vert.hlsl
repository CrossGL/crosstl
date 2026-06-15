
struct VertexInput {
    float2 inPosition: TEXCOORD0;
    float3 inColor: TEXCOORD1;
};
struct VertexOutput {
    float3 outFragColor: TEXCOORD0;
    float4 gl_Position: SV_POSITION;
};
// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.gl_Position = float4(input.inPosition, 0.0, 1.0);
    output.outFragColor = input.inColor;
    return output;
}
