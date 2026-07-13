#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
const vec3 lightDir = vec3(0.424, 0.566, 0.707);

layout(location = 0) in vec4 texcoord;
layout(location = 1) in vec3 frag_pos;
layout(binding = 1) uniform sampler2D tex;
// Fragment Shader
float linearToSrgb(float linear);

vec3 linearToSrgb(vec3 linear);

vec4 linearToSrgb(vec4 linear);

float linearToSrgb(float linear) {
    if ((linear <= 0.0031308)) {
        return (linear * 12.92);
    } else {
        return ((1.055 * pow(linear, (1.0 / 2.4))) - 0.055);
    }
}

vec3 linearToSrgb(vec3 linear) {
    return vec3(linearToSrgb(linear.r), linearToSrgb(linear.g), linearToSrgb(linear.b));
}

vec4 linearToSrgb(vec4 linear) {
    return vec4(linearToSrgb(linear.rgb), linear.a);
}

layout(location = 0) out vec4 fragColor;
void main() {
    vec4 uFragColor;
    vec3 dX = dFdx(frag_pos);
    vec3 dY = dFdy(frag_pos);
    vec3 normal = normalize(cross(dX, dY));
    float light = max(0.0, dot(lightDir, normal));
    uFragColor = linearToSrgb((light * texture(tex, texcoord.xy)));
    fragColor = uFragColor;
    return;
}
