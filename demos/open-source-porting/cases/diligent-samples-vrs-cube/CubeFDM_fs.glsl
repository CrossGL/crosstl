#version 450 core
#extension GL_EXT_fragment_invocation_density : require

uniform sampler2D g_Texture;

layout(std140) uniform g_Constants
{
    mat4x4 WorldViewProj;
    int    PrimitiveShadingRate;
    int    DrawMode; // 0 - Cube, 1 - Cube + Shading rate
};

layout(location=0) in  vec2 in_UV;
layout(location=0) out vec4 out_Color;

vec4 FragmentDensityToColor()
{
    float h   = (clamp(1.0 - 1.0 / float(gl_FragSizeEXT.x * gl_FragSizeEXT.y), 0.0, 1.0)) / 1.35;
    vec3  col = vec3(abs(h * 6.0 - 3.0) - 1.0, 2.0 - abs(h * 6.0 - 2.0), 2.0 - abs(h * 6.0 - 4.0));
    return vec4(clamp(col, vec3(0.0), vec3(1.0)), 1.0);
}

void main()
{
    vec4 Col = texture(g_Texture, in_UV);
    switch (DrawMode)
    {
        case 0: out_Color = Col; break;
        case 1: out_Color = (Col + FragmentDensityToColor()) * 0.5; break;
    }
}
