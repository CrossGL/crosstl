#include <hip/device_functions.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/math_functions.h>

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
  output.position = IdentifierNode(name = vec4)(input.position, 1.0);
  return output;
}

__device__ FragmentOutput main(FragmentInput input) {
  FragmentOutput output;
  float noise = IdentifierNode(name = perlinNoise)(input.uv);
  float height = (noise * 10.0);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) color =
      IdentifierNode(name = vec3)((height / 10.0), (1.0 - (height / 10.0)),
                                  0.0);
  output.color = IdentifierNode(name = vec4)(color, 1.0);
  return output;
}

__device__ float perlinNoise(VectorType(
    element_type = PrimitiveType(name = float, size_bits = None), size = 2) p) {
  return IdentifierNode(name = fract)(
      (IdentifierNode(name = sin)(IdentifierNode(name = dot)(
           p, IdentifierNode(name = vec2)(12.9898, 78.233))) *
       43758.5453));
}
