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

// Vertex Shader
#[vertex_shader]
pub fn
main(input : VertexInput) -> VertexOutput {
  let output : VertexOutput;
  (MemberAccessNode(object = output, member = uv) = input.texCoord);
  (MemberAccessNode(object = output, member = position) =
       Vec4<f32>::new (input.position, 1.0));
  return (output);
}

// Fragment Shader
#[fragment_shader]
pub fn main(input : FragmentInput) -> FragmentOutput {
  let output : FragmentOutput;
  (r = input.uv.x);
  (g = input.uv.y);
  (b = 0.5);
  (MemberAccessNode(object = output, member = color) =
       Vec4<f32>::new (r, g, b, 1.0));
  return (output);
}
