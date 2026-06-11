#version 450
layout(location = 0) in vec4 vA;
layout(location = 1) in float vB;
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    vec4 FragColor;
    FragColor = round(vA);
    FragColor *= round(vB);
    fragColor = FragColor;
    return;
}
