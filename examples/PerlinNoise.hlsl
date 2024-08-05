
float perlinNoise(float2 p) {
    return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
}

struct VSInput {
    float3 position : POSITION;
};

struct VSOutput {
   float4 position : SV_POSITION;
    float2 vUV : TEXCOORD0;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.vUV = input.position.xy * 10.0;
    output.position = float4(input.position, 1.0);
    return output;
}

struct PSInput {
    float2 vUV : TEXCOORD0;
};

struct PSOutput {
    float4 fragColor : SV_TARGET0;
};

PSOutput PSMain(PSInput input) {
    PSOutput output;
    float noise = perlinNoise(input.vUV);
    float height = noise * 10.0;
    float3 color = float3(height / 10.0, 1.0 - height / 10.0, 0.0);
    output.fragColor = float4(color, 1.0);
    return output;
}

