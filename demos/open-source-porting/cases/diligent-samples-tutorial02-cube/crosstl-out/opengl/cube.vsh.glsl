#version 450 core
in vec3 VSIn_Pos;
in vec4 VSIn_Color;
out vec4 PSIn_Color;
// Constant Buffers
layout(std140, binding = 0) uniform Constants {
    mat4 g_WorldViewProj;
};
// Vertex Shader
void main() {
    gl_Position = (vec4(VSIn_Pos, 1.0) * g_WorldViewProj);
    PSIn_Color = VSIn_Color;
}
