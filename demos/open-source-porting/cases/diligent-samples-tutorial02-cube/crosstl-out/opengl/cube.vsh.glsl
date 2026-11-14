#version 450 core
layout(location = 0) in vec3 VSIn_Pos;
layout(location = 1) in vec4 VSIn_Color;
layout(location = 13) out vec4 PSIn_Color;
// Constant Buffers
layout(std140, binding = 0) uniform Constants {
    mat4 g_WorldViewProj;
};
// Vertex Shader
void main() {
    gl_Position = (vec4(VSIn_Pos, 1.0) * g_WorldViewProj);
    PSIn_Color = VSIn_Color;
}
