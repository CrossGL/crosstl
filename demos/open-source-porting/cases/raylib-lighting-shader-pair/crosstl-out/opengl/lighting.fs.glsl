
#version 330
in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;
struct Light {
    int enabled;
    int type;
    vec3 position;
    vec3 target;
    vec4 color;
};

uniform sampler2D texture0;
// Constant Buffers
layout(std140) uniform Uniforms {
    vec4 colDiffuse;
    Light lights[4];
    vec4 ambient;
    vec3 viewPos;
};
// Fragment Shader
layout(location = 0) out vec4 out_fragColor;
void main() {
    vec4 finalColor;
    vec4 texelColor = texture(texture0, fragTexCoord);
    vec3 lightDot = vec3(0.0);
    vec3 normal = normalize(fragNormal);
    vec3 viewD = normalize((viewPos - fragPosition));
    vec3 specular = vec3(0.0);
    vec4 tint = (colDiffuse * fragColor);
    for (int i = 0; (i < 4); (++i)) {
        if ((lights[i].enabled == 1)) {
            vec3 light = vec3(0.0);
            if ((lights[i].type == 0)) {
                light = (-normalize((lights[i].target - lights[i].position)));
            }
            if ((lights[i].type == 1)) {
                light = normalize((lights[i].position - fragPosition));
            }
            float NdotL = max(dot(normal, light), 0.0);
            lightDot += (lights[i].color.rgb * NdotL);
            float specCo = 0.0;
            if ((NdotL > 0.0)) {
                specCo = pow(max(0.0, dot(viewD, reflect((-light), normal))), 16.0);
            }
            specular += specCo;
        }
    }
    finalColor = (texelColor * ((tint + vec4(specular, 1.0)) * vec4(lightDot, 1.0)));
    finalColor += ((texelColor * (ambient / 10.0)) * tint);
    finalColor = pow(finalColor, vec4((1.0 / 2.2)));
    out_fragColor = finalColor;
    return;
}
