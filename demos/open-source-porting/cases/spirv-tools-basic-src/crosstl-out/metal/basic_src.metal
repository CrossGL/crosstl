
#include <metal_stdlib>
using namespace metal;

struct gl_PerVertex {
    float4 gl_Position;
    float gl_PointSize [[mediump]];
    float gl_ClipDistance[8];
    float gl_CullDistance[8];
};
// Vertex Shader
vertex void vertex_main() {
    __attribute__((unused)) float4 _ua_position;
    __attribute__((unused)) float4 ANGLEXfbPosition;
    __attribute__((unused)) float4 gl_Position;
    __attribute__((unused)) float gl_PointSize;
    __attribute__((unused)) float gl_ClipDistance[8];
    __attribute__((unused)) float gl_CullDistance[8];
    gl_Position = _ua_position;
    return;
}
