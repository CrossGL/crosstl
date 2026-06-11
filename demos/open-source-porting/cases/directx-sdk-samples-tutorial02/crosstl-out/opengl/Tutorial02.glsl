#version 450 core
#ifdef GL_VERTEX_SHADER
layout(location = 0) in vec4 Pos;
#endif
#ifdef GL_VERTEX_SHADER
// Vertex Shader
void main() {
    gl_Position = Pos;
    return;
}

#endif
#ifdef GL_FRAGMENT_SHADER
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 1.0, 0.0, 1.0);
    return;
}

#endif
