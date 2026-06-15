#version 450 core
#ifdef GL_VERTEX_SHADER
out vec4 out_color;
layout(location = 5) out vec2 out_texCoord;
#endif
#ifdef GL_FRAGMENT_SHADER
in vec4 in_out_color;
layout(location = 5) in vec2 in_out_texCoord;
#endif
#ifdef GL_VERTEX_SHADER
layout(location = 0) in vec4 position;
#endif
#ifdef GL_VERTEX_SHADER
in vec4 color;
#endif
#ifdef GL_VERTEX_SHADER
layout(location = 5) in vec2 texCoord;
#endif
layout(binding = 0) uniform sampler2D Texture;
uniform mat4 MatrixTransform;
#ifdef GL_VERTEX_SHADER
// Vertex Shader
void main() {
    gl_Position = (position * MatrixTransform);
    out_color = color;
    out_texCoord = texCoord;
    return;
}

#endif
#ifdef GL_FRAGMENT_SHADER
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = (texture(Texture, in_out_texCoord) * in_out_color);
    return;
}

#endif
