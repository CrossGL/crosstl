
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
layout(location = 0) out vec4 out_color;
#endif
struct FragmentOutput {
  vec4 color;
};

// Constant Buffers
layout(std140, binding = 0) uniform TestBuffer {
  float values[4];
  vec3 colors[2];
};
#ifdef GL_VERTEX_SHADER
// Vertex Shader
void main() {
  uv = texCoord;
  float scale = (values[0] + values[1]);
  vec3 position_ = (position * scale);
  out_position = vec4(position_, 1.0);
  return;
}

#endif
#ifdef GL_FRAGMENT_SHADER
// Fragment Shader
void main() {
  vec3 color = colors[0];
  if ((in_uv.x > 0.5)) {
    color = colors[1];
  }
  out_color = vec4(color, 1.0);
  return;
}

#endif
