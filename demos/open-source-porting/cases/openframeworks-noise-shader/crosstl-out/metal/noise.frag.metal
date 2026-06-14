
#include <metal_stdlib>
using namespace metal;

// Constant Buffers
struct Uniforms {
    float4 globalColor;
};
// Fragment Shader
fragment float4 fragment_main(float4 _crossglFragCoord [[position]], constant Uniforms& uniforms [[buffer(0)]], texture2d<float> src_tex_unit0 [[texture(0)]]) {
    float4 fragColor;
    float xVal = _crossglFragCoord.x;
    float yVal = _crossglFragCoord.y;
    if (((xVal) - ((2.0) * floor((xVal) / (2.0)))) == 0.5 && ((yVal) - ((4.0) * floor((yVal) / (4.0)))) == 0.5) {
        fragColor = uniforms.globalColor;
    } else {
        discard_fragment();
    }
    return fragColor;
}
