#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
layout(location = 0) out vec4 texcoord;
layout(location = 1) out vec3 frag_pos;
// Constant Buffers
layout(std140, binding = 0) uniform buf {
    mat4 MVP;
    vec4 position[36];
    vec4 attr[36];
};
// Vertex Shader
void main() {
    texcoord = attr[gl_VertexID];
    gl_Position = (MVP * position[gl_VertexID]);
    frag_pos = gl_Position.xyz;
    return;
}
