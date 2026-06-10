
struct PSInput {
    float4 position: SV_POSITION;
    float4 color: Color;
};
// Vertex Shader
PSInput VSMain(float4 position : Position, float4 color : Color) {
    PSInput result;
    result.position = position;
    result.color = color;
    return result;
}

float4 PSMain(PSInput input): SV_TARGET {
    return input.color;
}

// Fragment Shader
void PSMain_2() {
}
