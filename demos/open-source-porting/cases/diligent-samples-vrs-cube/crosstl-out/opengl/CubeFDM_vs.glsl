#version 450 core
layout(location = 0) in vec3 in_Pos;
layout(location = 1) in vec2 in_UV;
layout(location = 0) out vec2 out_UV;
// Constant Buffers
layout(std140, binding = 0) uniform g_Constants {
    mat4 WorldViewProj;
    int PrimitiveShadingRate;
    int DrawMode;
};
// Vertex Shader
void main() {
    gl_Position = (vec4(in_Pos, 1.0) * WorldViewProj);
    out_UV = in_UV;
    return;
}
