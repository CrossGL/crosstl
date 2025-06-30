#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
  return ['output'];
}

__device__ FragmentOutput main(FragmentInput input) {
  FragmentOutput output;
  float r;
  float g;
  float b;
  return ['output'];
}
