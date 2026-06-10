
#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
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
