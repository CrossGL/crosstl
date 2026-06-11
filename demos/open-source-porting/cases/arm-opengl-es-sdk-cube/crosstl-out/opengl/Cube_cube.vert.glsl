#version 450 core
in vec4 av4position;
in vec3 av3colour;
out vec3 vv3colour;
// Constant Buffers
layout(std140, binding = 0) uniform Uniforms {
    mat4 mvp;
};
// Vertex Shader
void main() {
    vv3colour = av3colour;
    gl_Position = (mvp * av4position);
    return;
}
