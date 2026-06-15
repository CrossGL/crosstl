#version 330 core
in vec3 color;
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = vec4(color, 1.0);
    return;
}
