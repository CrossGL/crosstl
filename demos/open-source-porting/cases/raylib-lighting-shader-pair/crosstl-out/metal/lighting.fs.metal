
#include <metal_stdlib>
using namespace metal;

struct Light {
    int enabled;
    int type;
    float3 position;
    float3 target;
    float4 color;
};
struct FragmentInput {
    float3 fragPosition;
    float2 fragTexCoord;
    float4 fragColor;
    float3 fragNormal;
};
// Constant Buffers
struct Uniforms {
    float4 colDiffuse;
    Light lights[4];
    float4 ambient;
    float3 viewPos;
};
// Fragment Shader
fragment float4 fragment_main(FragmentInput input [[stage_in]], constant Uniforms& uniforms [[buffer(0)]], texture2d<float> texture0 [[texture(0)]]) {
    float4 finalColor;
    float4 texelColor = texture0.sample(sampler(mag_filter::linear, min_filter::linear), input.fragTexCoord);
    float3 lightDot = float3(0.0);
    float3 normal = normalize(input.fragNormal);
    float3 viewD = normalize(uniforms.viewPos - input.fragPosition);
    float3 specular = float3(0.0);
    float4 tint = uniforms.colDiffuse * input.fragColor;
    for (int i = 0; i < 4; ++i) {
        if (uniforms.lights[i].enabled == 1) {
            float3 light = float3(0.0);
            if (uniforms.lights[i].type == 0) {
                light = -normalize(uniforms.lights[i].target - uniforms.lights[i].position);
            }
            if (uniforms.lights[i].type == 1) {
                light = normalize(uniforms.lights[i].position - input.fragPosition);
            }
            float NdotL = max(dot(normal, light), 0.0);
            lightDot += uniforms.lights[i].color.rgb * float3(NdotL);
            float specCo = 0.0;
            if (NdotL > 0.0) {
                specCo = pow(max(0.0, dot(viewD, reflect(-light, normal))), 16.0);
            }
            specular += specCo;
        }
    }
    finalColor = texelColor * tint + float4(specular, 1.0) * float4(lightDot, 1.0);
    finalColor += texelColor * float4(uniforms.ambient / 10.0) * tint;
    finalColor = pow(finalColor, float4(1.0 / 2.2));
    return finalColor;
}

