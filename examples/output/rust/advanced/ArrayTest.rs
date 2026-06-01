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

// Constant Buffers
// CrossGL resource metadata: name=TestBuffer kind=cbuffer set=0 binding=0
// binding_source=automatic
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TestBuffer {
  pub values : [f32; 4], pub colors : [Vec3<f32>; 2],
}

static VALUES : [f32; 4] = [0.0; 4];
static COLORS : [Vec3<f32>; 2] = unsafe{std::mem::zeroed()};
// Vertex Shader
#[cfg_attr(feature = "crossgl_gpu", vertex_shader)]
pub fn vertex_main(input : VertexInput) -> VertexOutput {
  let mut output : VertexOutput = Default::default();
  output.uv = input.texCoord;
  let scale : f32 = (VALUES[0] + VALUES[1]);
  let position : Vec3<f32> = Vec3::<f32>::new ((input.position.x * scale),
                                               (input.position.y * scale),
                                               (input.position.z * scale));
  output.position = Vec4::<f32>::new (position.x, position.y, position.z, 1.0);
  return output;
}

// Fragment Shader
#[cfg_attr(feature = "crossgl_gpu", fragment_shader)]
pub fn fragment_main(input : FragmentInput) -> FragmentOutput {
  let mut output : FragmentOutput = Default::default();
  let mut color : Vec3<f32> = COLORS[0];
  if (input.uv.x > 0.5) {
    color = COLORS[1];
  }
  output.color = Vec4::<f32>::new (color.x, color.y, color.z, 1.0);
  return output;
}
