
#version 450 core
#ifdef GL_VERTEX_SHADER
layout(location = 4) out vec2 out_uv;
#endif
#ifdef GL_FRAGMENT_SHADER
layout(location = 4) in vec2 in_out_uv;
#endif
#ifdef GL_VERTEX_SHADER
layout(location = 0) in vec4 position;
#endif
#ifdef GL_VERTEX_SHADER
layout(location = 4) in vec2 uv;
#endif
layout(binding = 0) uniform sampler2D g_texture;
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
    fragColor = texture(g_texture, in_out_uv);
    return;
}

#endif
