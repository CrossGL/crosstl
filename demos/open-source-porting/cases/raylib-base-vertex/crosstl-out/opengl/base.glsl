#version 330
#if defined(GL_SPIRV) || defined(VULKAN)
#if !defined(GL_ES) && __VERSION__ < 410
#extension GL_ARB_separate_shader_objects : require
#endif
#define CROSSGL_LOCATION(n) layout(location = n)
#else
#define CROSSGL_LOCATION(n)
#endif
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec2 vertexTexCoord;
layout(location = 2) in vec3 vertexNormal;
layout(location = 3) in vec4 vertexColor;
CROSSGL_LOCATION(0) out vec2 fragTexCoord;
CROSSGL_LOCATION(1) out vec4 fragColor;
// Constant Buffers
layout(std140) uniform Uniforms {
    mat4 mvp;
};
// Vertex Shader
void main() {
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    gl_Position = (mvp * vec4(vertexPosition, 1.0));
    return;
}
