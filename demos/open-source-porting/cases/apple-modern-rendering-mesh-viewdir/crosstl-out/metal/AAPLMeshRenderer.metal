
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
struct main_vertex_Input {
    constant Camera& camera [[buffer(0)]];
};

vertex Output main_vertex(Input in_ [[stage_in]], main_vertex_Input _crossglInput [[stage_in]]) {
    constant Camera& camera = _crossglInput.camera;
    return Output() /* fallback for unmatched generated control flow */;
}
