// Generated Rust GPU Shader Code
use gpu::*;
use math::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VertexInput {
  pub position : Vec3<f32>, pub texCoord : Vec2<f32>,
}

impl VertexInput {
  pub fn new (position : Vec3<f32>, texCoord : Vec2<f32>)->Self {
    Self { position, texCoord }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VertexOutput {
  pub uv : Vec2<f32>, pub position : Vec4<f32>,
}

impl VertexOutput {
  pub fn new (uv : Vec2<f32>, position : Vec4<f32>)->Self {
    Self { uv, position }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FragmentInput {
  pub uv : Vec2<f32>,
}

impl FragmentInput {
  pub fn new (uv : Vec2<f32>)->Self {
    Self { uv }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FragmentOutput {
  pub color : Vec4<f32>,
}

impl FragmentOutput {
  pub fn new (color : Vec4<f32>)->Self {
    Self { color }
  }
}

// Vertex Shader
#[cfg_attr(feature = "crossgl_gpu", vertex_shader)]
pub fn vertex_main(input : VertexInput) -> VertexOutput {
  let mut output : VertexOutput = Default::default();
  output.uv = input.texCoord;
  output.position = Vec4::<f32>::new (input.position.x, input.position.y,
                                      input.position.z, 1.0);
  return output;
}

// Fragment Shader
#[cfg_attr(feature = "crossgl_gpu", fragment_shader)]
pub fn fragment_main(input : FragmentInput) -> FragmentOutput {
  let mut output : FragmentOutput = Default::default();
  let r : f32 = input.uv.x;
  let g : f32 = input.uv.y;
  let b : f32 = 0.5;
  output.color = Vec4::<f32>::new (r, g, b, 1.0);
  return output;
}
