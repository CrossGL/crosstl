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

// Constant Buffers
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TestBuffer {
  pub values : [f32; 4], pub colors : [Vec3<f32>; 2],
}

// Vertex Shader
#[vertex_shader]
pub fn
main(input : VertexInput) -> VertexOutput {
  let output : VertexOutput;
  (MemberAccessNode(object = output, member = uv) = input.texCoord);
  (scale = (values[0] + values[1]));
  (position = (input.position * scale));
  (MemberAccessNode(object = output, member = position) =
       Vec4<f32>::new (position, 1.0));
  return (output);
}

// Fragment Shader
#[fragment_shader]
pub fn main(input : FragmentInput) -> FragmentOutput {
  let output : FragmentOutput;
  (color = colors[0]);
  if (input.uv.x > 0.5) {
    (color = colors[1]);
  }
  (MemberAccessNode(object = output, member = color) =
       Vec4<f32>::new (color, 1.0));
  return (output);
}
