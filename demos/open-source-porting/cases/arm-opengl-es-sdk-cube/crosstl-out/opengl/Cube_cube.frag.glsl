#version 330 core
precision lowp float;
#if defined(GL_SPIRV) || defined(VULKAN)
#if !defined(GL_ES) && __VERSION__ < 410
#extension GL_ARB_separate_shader_objects : require
#endif
#define CROSSGL_LOCATION(n) layout(location = n)
#else
#define CROSSGL_LOCATION(n)
#endif
CROSSGL_LOCATION(0) in vec3 vv3colour;
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = vec4(vv3colour, 1.0);
    return;
}
