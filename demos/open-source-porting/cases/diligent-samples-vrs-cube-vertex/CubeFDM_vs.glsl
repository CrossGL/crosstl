#version 450 core

layout(std140) uniform g_Constants
{
    mat4x4 WorldViewProj;
    int    PrimitiveShadingRate;
    int    DrawMode; // 0 - Cube, 1 - Cube + Shading rate
};

layout(location=0) in vec3 in_Pos;
layout(location=1) in vec2 in_UV;

layout(location=0) out vec2 out_UV;

void main()
{
	gl_Position = vec4(in_Pos, 1.0) * WorldViewProj;
    out_UV      = in_UV;
}
