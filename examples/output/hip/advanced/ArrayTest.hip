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

ArrayType(element_type = PrimitiveType(name = float, size_bits = None),
          size = LiteralNode(value = 4,
                             literal_type = PrimitiveType(
                                 name = int, size_bits = None))) values;
ArrayType(
    element_type = VectorType(element_type = PrimitiveType(name = float,
                                                           size_bits = None),
                              size = 3),
    size = LiteralNode(value = 2,
                       literal_type = PrimitiveType(name = int,
                                                    size_bits = None))) colors;
__device__ VertexOutput main(VertexInput input) {
  VertexOutput output;
  output.uv = input.texCoord;
  float scale = (values[0] + values[1]);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) position = (input.position * scale);
  output.position = IdentifierNode(name = vec4)(position, 1.0);
  return output;
}

__device__ FragmentOutput main(FragmentInput input) {
  FragmentOutput output;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) color = colors[0];
  if ((input.uv.x > 0.5)) {
    color = colors[1];
  }
  output.color = IdentifierNode(name = vec4)(color, 1.0);
  return output;
}
