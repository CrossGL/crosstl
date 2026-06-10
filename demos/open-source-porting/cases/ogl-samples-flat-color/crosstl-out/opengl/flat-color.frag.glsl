
#version 330 core
precision highp float;
precision highp int;
// Constant Buffers
layout(std140) uniform Uniforms {
    vec4 Diffuse;
};
// Fragment Shader
layout(location = 0, index = 0) out vec4 fragColor;
void main() {
    vec4 Color;
    Color = Diffuse;
    fragColor = Color;
    return;
}
