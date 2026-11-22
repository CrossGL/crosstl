#version 450 core
#ifdef VULKAN
#define CROSSGL_SAMPLED(vulkan, opengl) vulkan
#else
#define CROSSGL_SAMPLED(vulkan, opengl) opengl
#endif
#ifdef GL_VERTEX_SHADER
layout(location = 5) out vec2 out_uv;
#endif
#ifdef GL_FRAGMENT_SHADER
layout(location = 5) in vec2 out_uv;
#endif
#ifdef GL_VERTEX_SHADER
layout(location = 0) in vec4 position;
#endif
#ifdef GL_VERTEX_SHADER
layout(location = 5) in vec2 uv;
#endif
#ifdef VULKAN
layout(binding = 0) uniform texture2D g_texture;
#else
layout(binding = 0) uniform sampler2D g_texture;
#endif
#ifdef VULKAN
layout(binding = 1) uniform sampler g_sampler;
#endif
#ifdef GL_VERTEX_SHADER
// Vertex Shader
void main() {
    gl_Position = position;
    out_uv = uv;
    return;
}

#endif
#ifdef GL_FRAGMENT_SHADER
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = texture(CROSSGL_SAMPLED(sampler2D(g_texture, g_sampler), g_texture), out_uv);
    return;
}

#endif
