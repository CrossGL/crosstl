#version 330 core
#if defined(GL_SPIRV) || defined(VULKAN)
#if !defined(GL_ES) && __VERSION__ < 410
#extension GL_ARB_separate_shader_objects : require
#endif
#define CROSSGL_LOCATION(n) layout(location = n)
#else
#define CROSSGL_LOCATION(n)
#endif
layout(location = 0) in vec4 a_position;
layout(location = 1) in vec2 a_texCoord;
CROSSGL_LOCATION(0) out vec2 v_texCoord;
// Vertex Shader
void main() {
    gl_Position = a_position;
    v_texCoord = a_texCoord;
    return;
}
