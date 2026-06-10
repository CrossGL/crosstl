
#version 450
layout(location = 0) in vec3 nearPoint;
layout(location = 1) in vec3 farPoint;
// Fragment Shader
vec4 grid(vec3 pos) {
    vec2 coord = pos.xz;
    vec2 derivative = fwidth(coord);
    vec2 gridLine = (abs((fract((coord - 0.5)) - 0.5)) / derivative);
    float line = min(gridLine.x, gridLine.y);
    return vec4(0.5, 0.5, 0.5, (1.0 - min(line, 1.0)));
}

layout(location = 0) out vec4 fragColor;
void main() {
    vec4 outColor;
    float t = ((-nearPoint.y) / (farPoint.y - nearPoint.y));
    vec3 pos = (nearPoint + (t * (farPoint - nearPoint)));
    outColor = grid(pos);
    fragColor = outColor;
    return;
}
