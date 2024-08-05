#version 450


// Vertex shader

float perlinNoise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

layout(location = 0) in vec3 position;
out vec2 vUV;

void main() {
    vUV = position.xy * 10.0;
    gl_Position = vec4(position, 1.0);
}

// Fragment shader

in vec2 vUV;
layout(location = 0) out vec4 fragColor;

void main() {
    float noise = perlinNoise(vUV);
    float height = noise * 10.0;
    vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
    fragColor = vec4(color, 1.0);
}
