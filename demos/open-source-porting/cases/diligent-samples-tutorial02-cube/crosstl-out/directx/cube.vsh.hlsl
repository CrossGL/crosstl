
// Constant Buffers
cbuffer Constants : register(b0) {
    float4x4 g_WorldViewProj;
};
// Vertex Shader
void VSMain(in float3 VSIn_Pos : ATTRIB0, in float4 VSIn_Color : ATTRIB1, out float4 PSIn_Pos : SV_POSITION, out float4 PSIn_Color : Color0) {
    PSIn_Pos = mul(float4(VSIn_Pos, 1.0), g_WorldViewProj);
    PSIn_Color = VSIn_Color;
}
