#version 450
layout(location = 0) in vec3 inFragColor;
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    vec4 outColor;
    outColor = vec4(inFragColor, 1.0);
    fragColor = outColor;
    return;
}
