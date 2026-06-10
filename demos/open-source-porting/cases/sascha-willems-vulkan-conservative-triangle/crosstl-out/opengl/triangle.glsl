
#version 450
out gl_PerVertex {
    vec4 gl_Position;
};
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 outColor;
struct UBO {
    mat4 projection;
    mat4 model;
};

// Constant Buffers
layout(std140, binding = 0) uniform Uniforms {
    UBO ubo;
};
// Vertex Shader
void main() {
    outColor = inColor;
    gl_Position = ((ubo.projection * ubo.model) * vec4(inPos, 1.0));
    return;
}
