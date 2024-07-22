#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float3 position [[attribute(0)]];
};

struct FragmentOutput {
    float4 color [[color(0)]];
};

fragment FragmentOutput frmain(VertexInput input [[stage_in]]) {
    FragmentOutput output;
    output.color = float4(input.position, 1.0);
    return output;
}

