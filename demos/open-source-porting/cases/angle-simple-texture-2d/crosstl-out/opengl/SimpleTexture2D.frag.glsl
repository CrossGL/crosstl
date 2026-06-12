#version 450 core
precision mediump float;
in vec2 v_texCoord;
layout(binding = 0) uniform sampler2D s_texture;
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = texture(s_texture, v_texCoord);
    return;
}
