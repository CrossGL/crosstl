#version 450 core
#ifdef GL_VERTEX_SHADER
layout(location = 13) out vec4 out_color;
#endif
#ifdef GL_FRAGMENT_SHADER
layout(location = 13) in vec4 in_out_color;
#endif
#ifdef GL_VERTEX_SHADER
layout(location = 0) in vec4 position;
#endif
#ifdef GL_VERTEX_SHADER
layout(location = 13) in vec4 color;
#endif
// Constant Buffers
layout(std140, binding = 0) uniform SceneConstantBuffer {
    vec4 offset;
    vec4 padding[15];
};
#ifdef GL_VERTEX_SHADER
// Vertex Shader
void main() {
    gl_Position = (position + offset);
    out_color = color;
    return;
}

#endif
#ifdef GL_FRAGMENT_SHADER
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = in_out_color;
    return;
}

#endif
