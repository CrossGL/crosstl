
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
  float3 position;
};
struct VertexOutput {
  float2 uv;
  float4 position;
};
struct FragmentInput {
  float2 uv;
};
struct FragmentOutput {
  float4 color;
};
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]]) {
  VertexOutput output;
  output.uv = input.position.xy * 10.0;
  output.position = float4(input.position, 1.0);
  return output;
}

// Fragment Shader
fragment FragmentOutput fragment_main(FragmentInput input [[stage_in]]) {
  FragmentOutput output;
  float noise = perlinNoise(input.uv);
  float3 color = float3(height / 10.0, 1.0 - height / 10.0, 0.0);
  return output;
}
