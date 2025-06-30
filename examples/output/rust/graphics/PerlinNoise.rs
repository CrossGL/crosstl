// Generated Rust GPU Shader Code
use gpu::*;
use math::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexInput {
  pub position : Vec3<f32>,
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
  output.uv = (input.position.xy * 10.0);
  output.position = Vec4<f32>::new (input.position, 1.0);
  return output;
}

// Fragment Shader
#[fragment_shader]
pub fn main(input : FragmentInput) -> FragmentOutput {
  let mut output : FragmentOutput;
  let mut noise : f32 = perlinNoise(input.uv);
  let mut height : f32 = (noise * 10.0);
  let mut color : Vec3<f32> = Vec3<f32>::new ((height / 10.0),
                                              (1.0 - (height / 10.0)), 0.0);
  output.color = Vec4<f32>::new (color, 1.0);
  return output;
}

pub fn perlinNoise(p : Vec2<f32>) -> f32 {
  return fract((sin(dot(p, Vec2<f32>::new (12.9898, 78.233))) * 43758.5453));
}
