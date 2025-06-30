
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
// Vertex Shader
void main() {
  VertexOutput output;
  output.uv = input.texCoord;
  output.position = IdentifierNode(name = vec4)(input.position, 1.0);
  return output;
}

// Fragment Shader
void main() {
  FragmentOutput output;
  float r = input.uv.x;
  float g = input.uv.y;
  float b = 0.5;
  output.color = IdentifierNode(name = vec4)(r, g, b, 1.0);
  return output;
}
