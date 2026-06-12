#version 450 core
in vec4 a_position;
in vec4 a_color;
in vec2 a_texCoord0;
out vec4 v_color;
out vec2 v_texCoords;
// Constant Buffers
layout(std140, binding = 0) uniform Uniforms {
    mat4 u_projTrans;
};
// Vertex Shader
void main() {
    v_color = a_color;
    v_texCoords = a_texCoord0;
    gl_Position = (u_projTrans * a_position);
    return;
}
