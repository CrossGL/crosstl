
#version 450 core
struct VertexInput {
  vec3 position;
  vec2 texCoord;
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
layout(std140, binding = 0)
    ArrayType(element_type = PrimitiveType(name = float, size_bits = None),
              size = LiteralNode(value = 4,
                                 literal_type = PrimitiveType(
                                     name = int, size_bits = None))) values;
layout(std140, binding = 1) ArrayType(
    element_type = VectorType(element_type = PrimitiveType(name = float,
                                                           size_bits = None),
                              size = 3),
    size = LiteralNode(value = 2,
                       literal_type = PrimitiveType(name = int,
                                                    size_bits = None))) colors;
// Vertex Shader
void main() {
  VertexOutput output;
  output.uv = input.texCoord;
  float scale = (values[0] + values[1]);
  vec3 position = (input.position * scale);
  output.position = IdentifierNode(name = vec4)(position, 1.0);
  return output;
}

// Fragment Shader
void main() {
  FragmentOutput output;
  vec3 color = colors[0];
  if ((input.uv.x > 0.5)) {
    color = colors[1];
  }
  output.color = IdentifierNode(name = vec4)(color, 1.0);
  return output;
}
