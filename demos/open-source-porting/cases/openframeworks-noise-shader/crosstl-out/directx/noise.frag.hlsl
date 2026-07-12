
Texture2D src_tex_unit0 : register(t0);
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4 globalColor;
};
// Fragment Shader
float4 PSMain(float4 _crossglFragCoord : SV_Position): SV_Target0 {
    float4 fragColor;
    float xVal = _crossglFragCoord.x;
    float yVal = _crossglFragCoord.y;
    if ((((xVal) - ((2.0) * floor((xVal) / (2.0)))) == 0.5) && (((yVal) - ((4.0) * floor((yVal) / (4.0)))) == 0.5)) {
        fragColor = globalColor;
    } else {
        discard;
    }
    return fragColor;
}
