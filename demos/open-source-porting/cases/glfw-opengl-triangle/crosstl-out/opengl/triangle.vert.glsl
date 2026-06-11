#version 330 core
in vec3 vCol;
in vec2 vPos;
out vec3 color;
// Constant Buffers
layout(std140) uniform Uniforms {
    mat4 MVP;
};
// Vertex Shader
void main() {
    gl_Position = (MVP * vec4(vPos, 0.0, 1.0));
    color = vCol;
    return;
}
