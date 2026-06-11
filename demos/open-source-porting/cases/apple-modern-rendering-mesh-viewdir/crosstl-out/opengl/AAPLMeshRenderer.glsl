#version 450 core
in vec3 position;
out vec3 viewDir;
layout(std140, binding = 0) uniform Camera {
    mat4 invViewMatrix;
} camera;
// Vertex Shader
void main() {
    viewDir = vec3(normalize((camera.invViewMatrix[3].xyz - position)));
    return;
}
