
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
  float3 position [[attribute(0)]];
  float2 texCoord [[attribute(1)]];
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
// Constant Buffers
struct TestBuffer {
  float values[4];
  float3 colors[2];
};
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]],
                                constant TestBuffer &testBuffer [[buffer(0)]]) {
  VertexOutput output;
  output.uv = input.texCoord;
  float scale = testBuffer.values[0] + testBuffer.values[1];
  float3 position = input.position * float3(scale);
  output.position = float4(position, 1.0);
  return output;
}

// Fragment Shader
fragment FragmentOutput fragment_main(FragmentInput input [[stage_in]],
                                      constant TestBuffer &testBuffer
                                      [[buffer(0)]]) {
  FragmentOutput output;
  float3 color = testBuffer.colors[0];
  if (input.uv.x > 0.5) {
    color = testBuffer.colors[1];
  }
  output.color = float4(color, 1.0);
  return output;
}
