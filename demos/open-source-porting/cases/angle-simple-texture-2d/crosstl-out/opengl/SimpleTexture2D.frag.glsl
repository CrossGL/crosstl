#version 450 core
precision mediump float;
#if defined(GL_SPIRV) || defined(VULKAN)
#if !defined(GL_ES) && __VERSION__ < 410
#extension GL_ARB_separate_shader_objects : require
#endif
#define CROSSGL_LOCATION(n) layout(location = n)
#else
#define CROSSGL_LOCATION(n)
#endif
CROSSGL_LOCATION(0) in vec2 v_texCoord;
layout(binding = 0) uniform sampler2D s_texture;
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = texture(s_texture, v_texCoord);
    return;
}
