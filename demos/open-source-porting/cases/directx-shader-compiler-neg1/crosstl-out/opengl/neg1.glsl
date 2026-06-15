#version 450 core
in vec4 a;
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = (-a.yxxx);
    return;
}
