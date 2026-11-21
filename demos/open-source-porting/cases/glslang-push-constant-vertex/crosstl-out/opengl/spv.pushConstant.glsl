#version 400
#if defined(GL_SPIRV) || defined(VULKAN)
#if !defined(GL_ES) && __VERSION__ < 410
#extension GL_ARB_separate_shader_objects : require
#endif
#define CROSSGL_LOCATION(n) layout(location = n)
#else
#define CROSSGL_LOCATION(n)
#endif
CROSSGL_LOCATION(0) out vec4 color;
// Constant Buffers
#ifdef VULKAN
layout(push_constant) uniform Material {
    int kind;
    float fa[3];
};
void crossgl_load_Material() {}
#else
layout(std140) uniform Material {
    uvec4 crossgl_cbuffer_registers[1];
};
int kind;
float fa[3];
uint crossgl_Material_word(int word) {
    return crossgl_cbuffer_registers[word >> 2][word & 3];
}
void crossgl_load_Material() {
    kind = int(crossgl_cbuffer_registers[0].x);
    for (int crossgl_index = 0; crossgl_index < 3; ++crossgl_index) {
        fa[crossgl_index] = uintBitsToFloat(crossgl_Material_word((crossgl_index) + 1));
    }
}
#endif
// Vertex Shader
void main() {
    crossgl_load_Material();
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
