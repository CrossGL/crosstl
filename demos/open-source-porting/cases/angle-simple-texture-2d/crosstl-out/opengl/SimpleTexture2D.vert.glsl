#version 450 core
in vec4 a_position;
in vec2 a_texCoord;
out vec2 v_texCoord;
// Vertex Shader
void main() {
    gl_Position = a_position;
    v_texCoord = a_texCoord;
    return;
}
