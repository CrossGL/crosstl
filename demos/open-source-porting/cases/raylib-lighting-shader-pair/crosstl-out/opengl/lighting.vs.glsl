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
CROSSGL_LOCATION(0) out vec3 fragPosition;
CROSSGL_LOCATION(1) out vec2 fragTexCoord;
CROSSGL_LOCATION(2) out vec4 fragColor;
CROSSGL_LOCATION(3) out vec3 fragNormal;
// Constant Buffers
layout(std140) uniform Uniforms {
    mat4 mvp;
    mat4 matModel;
    mat4 matNormal;
};
// Vertex Shader
void main() {
    fragPosition = vec3((matModel * vec4(vertexPosition, 1.0)));
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = normalize(vec3((matNormal * vec4(vertexNormal, 1.0))));
    gl_Position = (mvp * vec4(vertexPosition, 1.0));
    return;
}
