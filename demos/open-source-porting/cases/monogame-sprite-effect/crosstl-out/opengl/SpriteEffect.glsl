#version 450 core
out vec4 position;
out vec4 color;
out vec2 texCoord;
struct VSOutput {
    vec4 position;
    vec4 color;
    vec2 texCoord;
};

layout(binding = 0) uniform sampler2D Texture;
uniform mat4 MatrixTransform;
VSOutput SpriteVertexShader(vec4 position, vec4 color, vec2 texCoord) {
    VSOutput output_;
    output_.position = (position * MatrixTransform);
    output_.color = color;
    output_.texCoord = texCoord;
    return output_;
}

// Fragment Shader
vec4 SpritePixelShader(VSOutput input_) {
    return (texture(Texture, input_.texCoord) * input_.color);
}

void main() {
}
