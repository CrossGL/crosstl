#include <hip/device_functions.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/math_functions.h>

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

__device__ VertexOutput main(VertexInput input) {
  VertexOutput output;
  output.uv = input.texCoord;
  output.position = IdentifierNode(name = vec4)(input.position, 1.0);
  return output;
}

__device__ FragmentOutput main(FragmentInput input) {
  FragmentOutput output;
  float r = input.uv.x;
  float g = input.uv.y;
  float b = 0.5;
  output.color = IdentifierNode(name = vec4)(r, g, b, 1.0);
  return output;
}
