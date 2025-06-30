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

// Constant buffer: TestBuffer
__constant__ float[4] values;
__constant__ vec3[2] colors;

__device__ VertexOutput main(VertexInput input) {
  VertexOutput output;
  float scale;
  float3 position;
  return ['output'];
}

__device__ FragmentOutput main(FragmentInput input) {
  FragmentOutput output;
  float3 color;
  if ((input.uv.x > 0.5)) {
  }
  return ['output'];
}
