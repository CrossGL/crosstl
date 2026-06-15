
// Fragment Shader
void PSMain(in float4 PSIn_Pos : SV_Position, in float4 PSIn_Color : Color0, out float4 PSOut_Color : SV_TARGET) {
    float4 Color = PSIn_Color;
    PSOut_Color = Color;
}
