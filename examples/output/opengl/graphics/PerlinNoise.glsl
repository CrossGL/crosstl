#version 450 core
#ifdef GL_VERTEX_SHADER
in vec3 position;
#endif
#ifdef GL_VERTEX_SHADER
out vec2 uv;
out vec4 out_position;
#endif
#ifdef GL_FRAGMENT_SHADER
in vec2 in_uv;
#endif
#ifdef GL_FRAGMENT_SHADER
layout(location = 0) out vec4 out_color;
#endif
struct FragmentOutput {
  vec4 color;
};

#ifdef GL_VERTEX_SHADER
// Vertex Shader
void main() {
  uv = (position.xy * 10.0);
  out_position = vec4(position, 1.0);
  return;
}

#endif
#ifdef GL_FRAGMENT_SHADER
// Fragment Shader
float perlinNoise(vec2 p) {
  return fract((sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453));
}

void main() {
  float noise = perlinNoise(in_uv);
  float height = (noise * 10.0);
  vec3 color = vec3((height / 10.0), (1.0 - (height / 10.0)), 0.0);
  out_color = vec4(color, 1.0);
  return;
}

#endif
