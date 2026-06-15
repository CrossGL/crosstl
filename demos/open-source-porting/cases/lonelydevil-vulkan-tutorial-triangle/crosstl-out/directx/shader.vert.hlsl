
struct VertexOutput {
    float3 fragColor: TEXCOORD0;
    float4 gl_Position: SV_POSITION;
};
float2 positions[3];
float3 colors[3];
// Vertex Shader
VertexOutput VSMain(uint gl_VertexIndex : SV_VertexID) {
    VertexOutput output;
    output.gl_Position = float4(positions[gl_VertexIndex], 0.0, 1.0);
    output.fragColor = colors[gl_VertexIndex];
    return output;
}
