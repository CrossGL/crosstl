#version 450 core
in vec4 PSIn_Color;
layout(location = 0) out vec4 fragColor;
// Fragment Shader
void main() {
    vec4 Color = PSIn_Color;
    fragColor = Color;
}
