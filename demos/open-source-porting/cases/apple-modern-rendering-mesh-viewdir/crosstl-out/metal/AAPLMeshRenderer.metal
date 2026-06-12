
#include <metal_stdlib>
using namespace metal;

struct Camera {
    float4x4 invViewMatrix;
};
struct Input {
    float3 position [[attribute(0)]];
};
struct Output {
    half3 viewDir;
    /* CrossGL fallback: Metal vertex entry points require a position output even when the GLSL source did not write gl_Position. */
    float4 __crossgl_position [[position]];
};
// Vertex Shader
vertex Output main_vertex(Input in_ [[stage_in]], constant Camera& camera [[buffer(0)]]) {
    Output out_;
    out_.viewDir = half3(normalize(camera.invViewMatrix[3].xyz - in_.position));
    Output _crossglReturn = out_;
    _crossglReturn.__crossgl_position = float4(0.0, 0.0, 0.0, 1.0);
    return _crossglReturn;
}
