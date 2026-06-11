#version 400
out vec4 color;
// Constant Buffers
layout(std140) uniform Material {
    int kind;
    float fa[3];
};
// Vertex Shader
void main() {
    switch (kind) {
        case 1: {
            color = vec4(0.2);
            break;
        }
        case 2: {
            color = vec4(0.5);
            break;
        }
        default: {
            color = vec4(0.0);
            break;
        }
    }
    return;
}
