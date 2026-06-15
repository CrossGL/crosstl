
struct Light {
    int enabled;
    int type;
    float3 position;
    float3 target;
    float4 color;
};
struct FragmentInput {
    float3 fragPosition: TEXCOORD0;
    float2 fragTexCoord: TEXCOORD1;
    float4 fragColor: TEXCOORD2;
    float3 fragNormal: TEXCOORD3;
};
Texture2D texture0 : register(t0);
SamplerState texture0Sampler : register(s0);
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4 colDiffuse;
    Light lights[4];
    float4 ambient;
    float3 viewPos;
};
// Fragment Shader
float4 PSMain(FragmentInput input): SV_Target0 {
    float4 finalColor;
    float4 texelColor = texture0.Sample(texture0Sampler, input.fragTexCoord);
    float3 lightDot = float3(0.0, 0.0, 0.0);
    float3 normal = normalize(input.fragNormal);
    float3 viewD = normalize((viewPos - input.fragPosition));
    float3 specular = float3(0.0, 0.0, 0.0);
    float4 tint = (colDiffuse * input.fragColor);
    for (int i = 0; (i < 4); ++i) {
        if ((lights[i].enabled == 1)) {
            float3 light = float3(0.0, 0.0, 0.0);
            if ((lights[i].type == 0)) {
                light = -normalize((lights[i].target - lights[i].position));
            }
            if ((lights[i].type == 1)) {
                light = normalize((lights[i].position - input.fragPosition));
            }
            float NdotL = max(dot(normal, light), 0.0);
            lightDot += (lights[i].color.rgb * NdotL);
            float specCo = 0.0;
            if ((NdotL > 0.0)) {
                specCo = pow(max(0.0, dot(viewD, reflect(-light, normal))), 16.0);
            }
            specular += specCo;
        }
    }
    finalColor = (texelColor * ((tint + float4(specular, 1.0)) * float4(lightDot, 1.0)));
    finalColor += ((texelColor * (ambient / 10.0)) * tint);
    finalColor = pow(finalColor, float4((1.0 / 2.2), (1.0 / 2.2), (1.0 / 2.2), (1.0 / 2.2)));
    return finalColor;
}
