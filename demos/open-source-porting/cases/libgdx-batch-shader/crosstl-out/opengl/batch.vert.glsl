#version 450 core
layout(location = 0) in vec4 a_position;
layout(location = 1) in vec4 a_color;
layout(location = 2) in vec2 a_texCoord0;
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
