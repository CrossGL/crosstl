
#version 450 core
#ifdef GL_VERTEX_SHADER
out vec4 out_color;
#endif
#ifdef GL_VERTEX_SHADER
in vec4 position;
#endif
#ifdef GL_VERTEX_SHADER
in vec4 color;
#endif
#ifdef GL_VERTEX_SHADER
// Vertex Shader
void main() {
    gl_Position = position;
    out_color = color;
    return;
}

#endif
#ifdef GL_FRAGMENT_SHADER
// Fragment Shader
vec4 PSMain(PSInput input) {
    return input.color;
}

void main() {
}

#endif
