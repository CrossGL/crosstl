float perlinNoise(float2 p) {
    return fract((sin(dot(p, float2(12.9898, 78.233))) * 43758.5453));
}

struct Vertex_INPUT {
    float3 position : POSITION;
};

struct Vertex_OUTPUT {
   float4 position : SV_POSITION;
    float2 vUV : TEXCOORD0;
};

Vertex_OUTPUT VSMain(Vertex_INPUT input) {
    Vertex_OUTPUT output;
    output.vUV = (input.position.xy * 10.0);
    output.position = float4(input.position, 1.0);
    return output;
}

struct Fragment_INPUT {
    float2 vUV : TEXCOORD0;
};

struct Fragment_OUTPUT {
    float4 fragColor : SV_TARGET0;
};

Fragment_OUTPUT PSMain(Fragment_INPUT input) {
    Fragment_OUTPUT output;
    float noise = perlinNoise(input.vUV);
    float height = (noise * 10.0);
    float3 color = float3((height / 10.0), (1.0 - (height / 10.0)), 0.0);
    output.fragColor = float4(color, 1.0);
    return output;
}

    