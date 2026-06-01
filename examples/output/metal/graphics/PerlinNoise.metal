
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
  float3 position [[attribute(0)]];
};
struct VertexOutput {
  float2 uv;
  float4 position [[position]];
};
struct FragmentInput {
  float2 uv;
};
struct FragmentOutput {
  float4 color [[color(0)]];
};
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]]) {
  VertexOutput output;
  output.uv = input.position.xy * float2(10.0);
  output.position = float4(input.position, 1.0);
  return output;
}

float perlinNoise(float2 p) {
  return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
}

// Fragment Shader
fragment FragmentOutput fragment_main(FragmentInput input [[stage_in]]) {
  FragmentOutput output;
  float noise = perlinNoise(input.uv);
  float height = noise * 10.0;
  float3 color = float3(height / 10.0, 1.0 - height / 10.0, 0.0);
  output.color = float4(color, 1.0);
  return output;
}
