#version 330 core
precision highp float;
precision highp int;
layout(location = 0) in vec4 Position;
// Constant Buffers
layout(std140) uniform Uniforms {
    mat4 MVP;
};
// Vertex Shader
void main() {
    gl_Position = (MVP * Position);
    return;
}
