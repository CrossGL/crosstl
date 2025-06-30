
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
// Constant Buffers
TestBuffer {
  float values[4];
  float3 colors[2];
};
constant TestBuffer &TestBuffer [[buffer(0)]];
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]]) {
  VertexOutput output;
  output.uv = input.texCoord;
  float scale = values[0] + values[1];
  float3 position = input.position * scale;
  output.position = float4(position, 1.0);
  return output;
}

// Fragment Shader
fragment FragmentOutput fragment_main(FragmentInput input [[stage_in]]) {
  FragmentOutput output;
  float3 color = colors[0];
  if (input.uv.x > 0.5) {
    color = colors[1];
  }
  output.color = float4(color, 1.0);
  return output;
}
