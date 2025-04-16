
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
  float3 position;
  float2 texCoord;
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
  output;
  output.uv = input.texCoord;
  output.position = float4(input.position, 1.0);
  return output;
}

// Fragment Shader
fragment FragmentOutput fragment_main(FragmentInput input [[stage_in]]) {
  output;
  float r = input.uv.x;
  float g = input.uv.y;
  float b = 0.5;
  output.color = float4(r, g, b, 1.0);
  return output;
}
