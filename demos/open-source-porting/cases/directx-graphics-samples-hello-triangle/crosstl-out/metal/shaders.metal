
#include <metal_stdlib>
using namespace metal;

struct PSInput {
  float4 position [[position]];
  float4 color [[Color]];
};
// Vertex Shader
vertex PSInput VSMain(float4 position [[Position]], float4 color [[Color]]) {
  PSInput result;
  result.position = position;
  result.color = color;
  return result;
}

float4 PSMain(PSInput input) [[color(0)]] { return input.color; }

// Fragment Shader
fragment void fragment_main() {}
