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

struct TestBuffer {
  float[LiteralNode(value = 4, literal_type = PrimitiveType(
                                   name = int, size_bits = None))] values;
  vec3[LiteralNode(value = 2, literal_type = PrimitiveType(
                                  name = int, size_bits = None))] colors;
};

__device__ VertexOutput main(VertexInput input) {
  VertexOutput output;
  output.uv = input.texCoord;
  float scale;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) position;
  output.position = vec4(position, 1.0);
  return output;
}

__device__ FragmentOutput main(FragmentInput input) {
  FragmentOutput output;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) color;
  if ((input.uv.x > 0.5)) {
    color = colors[1];
  }
  output.color = vec4(color, 1.0);
  return output;
}
