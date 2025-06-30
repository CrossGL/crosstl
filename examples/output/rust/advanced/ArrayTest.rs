// Generated Rust GPU Shader Code
use gpu::*;
use math::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexInput {
  pub position : Vec3<f32>, pub texCoord : Vec2<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexOutput {
  pub uv : Vec2<f32>, pub position : Vec4<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FragmentInput {
  pub uv : Vec2<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FragmentOutput {
  pub color : Vec4<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TestBuffer {
  pub values : vecLiteralNode(value = 4,
                              literal_type = PrimitiveType(name = int,
                                                           size_bits = None)),
               pub colors
      : vec3LiteralNode(value = 2,
                        literal_type = PrimitiveType(name = int,
                                                     size_bits = None)),
}

// Vertex Shader
#[vertex_shader]
pub fn
main(input : VertexInput) -> VertexOutput {
  let mut output : VertexOutput;
  output.uv = input.texCoord;
  let mut scale : f32 = (values[0] + values[1]);
  let mut position : Vec3<f32> = (input.position * scale);
  output.position = Vec4<f32>::new (position, 1.0);
  return output;
}

// Fragment Shader
#[fragment_shader]
pub fn main(input : FragmentInput) -> FragmentOutput {
  let mut output : FragmentOutput;
  let mut color : Vec3<f32> = colors[0];
  if (input.uv.x > 0.5) {
    color = colors[1];
  }
  output.color = Vec4<f32>::new (color, 1.0);
  return output;
}
