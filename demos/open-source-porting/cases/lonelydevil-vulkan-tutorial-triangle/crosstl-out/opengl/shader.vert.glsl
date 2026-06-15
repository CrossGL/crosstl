#version 450
layout(location = 0) out vec3 fragColor;
uniform vec2 positions[3] = { vec2(0.0, (-0.5)), vec2(0.5, 0.5), vec2((-0.5), 0.5) };
uniform vec3 colors[3] = { vec3(0.5, 0.5, 0.0), vec3(0.0, 0.5, 0.5), vec3(0.5, 0.0, 0.5) };
// Vertex Shader
void main() {
    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
    fragColor = colors[gl_VertexID];
    return;
}
