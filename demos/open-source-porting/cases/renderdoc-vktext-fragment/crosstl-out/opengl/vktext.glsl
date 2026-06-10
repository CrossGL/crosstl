
#version 450 core
layout(location = 0) in vec4 tex;
layout(location = 1) in vec2 glyphuv;
layout(binding = 3) uniform sampler2D tex0;
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
    vec4 color_out;
    float text = 0.0;
    if (((((glyphuv.x >= 0.0) && (glyphuv.x <= 1.0)) && (glyphuv.y >= 0.0)) && (glyphuv.y <= 1.0))) {
        vec2 uv;
        uv.x = mix(tex.x, tex.z, glyphuv.x);
        uv.y = mix(tex.y, tex.w, glyphuv.y);
        text = texture(tex0, uv.xy).x;
    }
    color_out = vec4(vec3(text), clamp((text + 0.5), 0.0, 1.0));
    fragColor = color_out;
    return;
}
