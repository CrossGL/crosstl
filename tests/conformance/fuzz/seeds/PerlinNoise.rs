// Generated Rust GPU Shader Code
use gpu::*;
use math::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VertexInput {
  pub position : Vec3<f32>,
}

impl VertexInput {
  pub fn new (position : Vec3<f32>)->Self {
    Self { position }
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
  output.uv =
      Vec2::<f32>::new ((input.position.x * 10.0), (input.position.y * 10.0));
  output.position = Vec4::<f32>::new (input.position.x, input.position.y,
                                      input.position.z, 1.0);
  return output;
}

// Fragment Shader
#[cfg_attr(feature = "crossgl_gpu", fragment_shader)]
pub fn fragment_main(input : FragmentInput) -> FragmentOutput {
  let mut output : FragmentOutput = Default::default();
  let noise : f32 = perlinNoise(input.uv);
  let height : f32 = (noise * 10.0);
  let color : Vec3<f32> = Vec3::<f32>::new ((height / 10.0),
                                            (1.0 - (height / 10.0)), 0.0);
  output.color = Vec4::<f32>::new (color.x, color.y, color.z, 1.0);
  return output;
}

pub fn perlinNoise(p : Vec2<f32>) -> f32 {
  return fract((sin({
                  let __cgl_vec_arg_0 = Vec2::<f32>::new (12.9898, 78.233);
                  (p.x * __cgl_vec_arg_0.x) + (p.y * __cgl_vec_arg_0.y)
                }) *
                43758.5453));
}
