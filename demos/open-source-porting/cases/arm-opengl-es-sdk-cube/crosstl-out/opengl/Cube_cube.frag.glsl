#version 450 core
precision lowp float;
in vec3 vv3colour;
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = vec4(vv3colour, 1.0);
    return;
}
