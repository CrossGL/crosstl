#version 330 core
uniform sampler2D src_tex_unit0;
// Constant Buffers
layout(std140) uniform Uniforms {
    vec4 globalColor;
};
// Fragment Shader
layout(location = 0) out vec4 out_fragColor;
void main() {
    vec4 fragColor;
    float xVal = gl_FragCoord.x;
    float yVal = gl_FragCoord.y;
    if (((mod(xVal, 2.0) == 0.5) && (mod(yVal, 4.0) == 0.5))) {
        fragColor = globalColor;
    } else {
        discard;
    }
    out_fragColor = fragColor;
    return;
}
