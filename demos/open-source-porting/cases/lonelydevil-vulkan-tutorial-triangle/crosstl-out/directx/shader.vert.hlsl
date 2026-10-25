
struct VertexOutput {
    float3 fragColor: TEXCOORD0;
    float4 gl_Position: SV_POSITION;
};
static float2 positions[3] = {float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5)};
static float3 colors[3] = {float3(0.5, 0.5, 0.0), float3(0.0, 0.5, 0.5), float3(0.5, 0.0, 0.5)};
// Vertex Shader
VertexOutput VSMain(uint gl_VertexIndex : SV_VertexID) {
    VertexOutput output;
    output.gl_Position = float4(positions[gl_VertexIndex], 0.0, 1.0);
    output.fragColor = colors[gl_VertexIndex];
    return output;
}
