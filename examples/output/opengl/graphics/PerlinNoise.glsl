
#version 450 core
struct VertexInput {
  vec3 position;
};
struct VertexOutput {
  vec2 uv;
  vec4 position;
};
struct FragmentInput {
  vec2 uv;
};
struct FragmentOutput {
  vec4 color;
};
// Vertex Shader
void main() {
  VertexOutput output;
  output.uv = (input.position.xy * 10.0);
  output.position = vec4(input.position, 1.0);
  return output;
}

// Fragment Shader
void main() {
  FragmentOutput output;
  float noise = perlinNoise(input.uv);
  float height = (noise * 10.0);
  vec3 color = vec3((height / 10.0), (1.0 - (height / 10.0)), 0.0);
  output.color = vec4(color, 1.0);
  return output;
}

float perlinNoise(VectorType(element_type = PrimitiveType(name = float,
                                                          size_bits = None),
                             size = 2) p) {
  return fract((sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453));
}
