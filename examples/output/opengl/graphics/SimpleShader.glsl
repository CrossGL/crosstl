
#version 450 core
#ifdef GL_VERTEX_SHADER
in vec3 position;
in vec2 texCoord;
#endif
#ifdef GL_VERTEX_SHADER
out vec2 uv;
out vec4 out_position;
#endif
#ifdef GL_FRAGMENT_SHADER
in vec2 in_uv;
#endif
#ifdef GL_FRAGMENT_SHADER
layout(location = 0) out vec4 color;
#endif
struct FragmentOutput {
  vec4 color;
};

#ifdef GL_VERTEX_SHADER
// Vertex Shader
void main() {
  uv = texCoord;
  out_position = vec4(position, 1.0);
  return;
}

#endif
#ifdef GL_FRAGMENT_SHADER
// Fragment Shader
void main() {
  float r = in_uv.x;
  float g = in_uv.y;
  float b = 0.5;
  color = vec4(r, g, b, 1.0);
  return;
}

#endif
