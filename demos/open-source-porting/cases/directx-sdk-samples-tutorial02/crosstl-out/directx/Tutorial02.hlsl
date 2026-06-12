
// Vertex Shader
float4 VSMain(float4 Pos : Position): SV_POSITION {
    return Pos;
}

// Fragment Shader
float4 PSMain(float4 Pos : SV_Position): SV_TARGET {
    return float4(1.0, 1.0, 0.0, 1.0);
}
