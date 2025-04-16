
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
  output;
  output.uv = input.position.xy * 10.0;
  output.position = vec4(input.position, 1.0);
  return output;
}

// Fragment Shader
void main() {
  output;
  float noise = perlinNoise(input.uv);
  vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
  return output;
}
