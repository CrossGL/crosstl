#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

__device__ VertexOutput main(VertexInput input) {
  VertexOutput output;
  output.uv = (input.position.xy * 10.0);
  output.position = vec4(input.position, 1.0);
  return output;
}

__device__ FragmentOutput main(FragmentInput input) {
  FragmentOutput output;
  float noise;
  float height;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) color;
  output.color = vec4(color, 1.0);
  return output;
}
