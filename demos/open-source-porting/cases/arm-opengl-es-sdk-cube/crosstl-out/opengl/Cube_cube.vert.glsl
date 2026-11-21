#version 420 core
#if defined(GL_SPIRV) || defined(VULKAN)
#if !defined(GL_ES) && __VERSION__ < 410
#extension GL_ARB_separate_shader_objects : require
#endif
#define CROSSGL_LOCATION(n) layout(location = n)
#else
#define CROSSGL_LOCATION(n)
#endif
layout(location = 0) in vec4 av4position;
layout(location = 1) in vec3 av3colour;
CROSSGL_LOCATION(0) out vec3 vv3colour;
// Constant Buffers
layout(std140, binding = 0) uniform Uniforms {
    mat4 mvp;
};
// Vertex Shader
void main() {
    vv3colour = av3colour;
    gl_Position = (mvp * av4position);
    return;
}
