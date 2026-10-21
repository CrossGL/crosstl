#version 450
out gl_PerVertex {
    vec4 gl_Position;
};
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 outColor;
// Constant Buffers
layout(std140, binding = 0) uniform UBO {
    mat4 projection;
    mat4 model;
};
// Vertex Shader
void main() {
    outColor = inColor;
    gl_Position = ((projection * model) * vec4(inPos, 1.0));
    return;
}
