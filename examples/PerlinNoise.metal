#include <metal_stdlib>
using namespace metal;

float perlinNoise(float2 p) {
    return fract((sin(dot(p, float2(12.9898, 78.233))) * 43758.5453));
}

struct Vertex_INPUT {
    float3 position [[attribute(0)]];
};

struct Vertex_OUTPUT {
    float4 position [[position]];
    float2 vUV;
};

vertex Vertex_OUTPUT vertex_main(Vertex_INPUT input [[stage_in]]) {
    Vertex_OUTPUT output;
    output.vUV = (input.position.xy * 10.0);
    output.position = float4(input.position, 1.0);
    return output;
}

struct Fragment_INPUT {
    float2 vUV [[stage_in]];
};

struct Fragment_OUTPUT {
    float4 fragColor [[color(0)]];
};

fragment Fragment_OUTPUT fragment_main(Fragment_INPUT input [[stage_in]]) {
    Fragment_OUTPUT output;
    float noise = perlinNoise(input.vUV);
    float height = (noise * 10.0);
    float3 color = float3((height / 10.0), (1.0 - (height / 10.0)), 0.0);
    output.fragColor = float4(color, 1.0);
    return output;
}
