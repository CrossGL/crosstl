#version 330
#if defined(GL_SPIRV) || defined(VULKAN)
#if !defined(GL_ES) && __VERSION__ < 410
#extension GL_ARB_separate_shader_objects : require
#endif
#define CROSSGL_LOCATION(n) layout(location = n)
#else
#define CROSSGL_LOCATION(n)
#endif
CROSSGL_LOCATION(0) in vec2 fragTexCoord;
CROSSGL_LOCATION(1) in vec4 fragColor;
uniform sampler2D texture0;
// Constant Buffers
layout(std140) uniform Uniforms {
    vec4 colDiffuse;
};
// Fragment Shader
layout(location = 0) out vec4 out_fragColor;
void main() {
    vec4 finalColor;
    vec4 texelColor = texture(texture0, fragTexCoord);
    finalColor = ((texelColor * colDiffuse) * fragColor);
    out_fragColor = finalColor;
    return;
}
