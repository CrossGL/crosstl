#version 450 core
in vec4 v_color;
in vec2 v_texCoords;
layout(binding = 0) uniform sampler2D u_texture;
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = (v_color * texture(u_texture, v_texCoords));
    return;
}
