#version 450 core
#ifdef GL_VERTEX_SHADER
in vec2 pos;
#endif
#ifdef GL_FRAGMENT_SHADER
layout(location = 0) out vec4 fragColor;
#endif
vec3 tonemap(vec3 color) {
    return color;
}

#ifdef GL_FRAGMENT_SHADER
// Fragment Shader
void main() {
    vec3 color = vec3(1.0, 0.5, 0.25);
    fragColor = vec4(tonemap(color), 1.0);
}

#endif
#ifdef GL_VERTEX_SHADER
// Vertex Shader
void main() {
    gl_Position = vec4(vec3(pos, 0.0), 1.0);
}

#endif
