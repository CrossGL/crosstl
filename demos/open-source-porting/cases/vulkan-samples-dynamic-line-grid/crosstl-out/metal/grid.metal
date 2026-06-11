
#include <metal_stdlib>
using namespace metal;

struct FragmentInput {
    float3 nearPoint [[attribute(0)]];
    float3 farPoint [[attribute(1)]];
};
float4 grid(float3 pos) {
    float2 coord = pos.xz;
    float2 derivative = fwidth(coord);
    float2 gridLine = abs(fract(coord - float2(0.5)) - 0.5) / derivative;
    float line = min(gridLine.x, gridLine.y);
    return float4(0.5, 0.5, 0.5, 1.0 - min(line, 1.0));
}

// Fragment Shader
fragment float4 fragment_main(FragmentInput input [[stage_in]]) {
    float4 outColor;
    float t = -input.nearPoint.y / (input.farPoint.y - input.nearPoint.y);
    float3 pos = input.nearPoint + float3(t) * (input.farPoint - input.nearPoint);
    outColor = grid(pos);
    return outColor;
}
