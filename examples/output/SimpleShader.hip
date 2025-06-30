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
  MemberAccessNode(object = output, member = uv);
  MemberAccessNode(object = output, member = position);
  return;
}

__device__ FragmentOutput main(FragmentInput input) {
  FragmentOutput output;
  float r;
  float g;
  float b;
  MemberAccessNode(object = output, member = color);
  return;
}
