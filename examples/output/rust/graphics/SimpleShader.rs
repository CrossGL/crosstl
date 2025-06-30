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
  let mut output : VertexOutput;
  output.uv = input.texCoord;
  output.position = Vec4<f32>::new (input.position, 1.0);
  return output;
}

// Fragment Shader
#[fragment_shader]
pub fn main(input : FragmentInput) -> FragmentOutput {
  let mut output : FragmentOutput;
  let mut r : f32 = input.uv.x;
  let mut g : f32 = input.uv.y;
  let mut b : f32 = 0.5;
  output.color = Vec4<f32>::new (r, g, b, 1.0);
  return output;
}
