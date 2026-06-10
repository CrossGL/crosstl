
#version 450
layout(location = 0) in vec3 fragColor;
// Fragment Shader
layout(location = 0) out vec4 out_fragColor;
void main() {
    vec4 outColor;
    outColor = vec4(fragColor, 1.0);
    out_fragColor = outColor;
    return;
}
