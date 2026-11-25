#version 330 core
#if defined(GL_SPIRV) || defined(VULKAN)
#if !defined(GL_ES) && __VERSION__ < 410
#extension GL_ARB_separate_shader_objects : require
#endif
#define CROSSGL_LOCATION(n) layout(location = n)
#else
#define CROSSGL_LOCATION(n)
#endif
layout(location = 0) in vec3 vCol;
layout(location = 1) in vec2 vPos;
CROSSGL_LOCATION(0) out vec3 color;
// Constant Buffers
layout(std140) uniform Uniforms {
    mat4 MVP;
};
// Vertex Shader
void main() {
    gl_Position = (MVP * vec4(vPos, 0.0, 1.0));
    color = vCol;
    return;
}
