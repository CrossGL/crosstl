// Generated Rust GPU Shader Code
use gpu::*;
use math::*;

#[derive(Debug, Clone, Copy, Default)]
pub enum Option<T> {
    Some(T),
#[default]
    None,
}

#[derive(Debug, Clone, Copy)]
pub enum Result<T, E> {
    Ok(T),
    Err(E),
}

impl<T: Default, E: Default> Default for Result<T, E> {
  fn default()->Self { Self::Ok(Default::default()) }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3<T>{
  pub x : T,
  pub y : T,
  pub z : T,
}

impl<T>
    Vec3<T> {
  pub fn new (x : T, y : T, z : T)->Self {
    Self { x, y, z }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Matrix3x3<T>{
  pub row0 : Vec3<T>,
  pub row1 : Vec3<T>,
  pub row2 : Vec3<T>,
}

impl<T>
    Matrix3x3<T> {
  pub fn new (row0 : Vec3<T>, row1 : Vec3<T>, row2 : Vec3<T>)->Self {
    Self { row0, row1, row2 }
  }
}

pub trait Numeric : Sized {
  fn add(self, other : Self)->Self;
  fn mul(self, other : Self)->Self;
  fn zero() -> Self;
  fn one() -> Self;
}

impl Numeric for f32 {
  fn add(self, other : Self) -> Self{self + other} fn mul(self, other : Self)
      ->Self{self * other} fn zero()
      ->Self{0.0} fn one()
      ->Self{1.0}
}

impl Numeric for f64 {
  fn add(self, other : Self) -> Self{self + other} fn mul(self, other : Self)
      ->Self{self * other} fn zero()
      ->Self{0.0} fn one()
      ->Self{1.0}
}

impl Numeric for i32 {
  fn add(self, other : Self) -> Self{self + other} fn mul(self, other : Self)
      ->Self{self * other} fn zero()
      ->Self{0} fn one()
      ->Self{1}
}

impl Numeric for u32 {
  fn add(self, other : Self) -> Self{self + other} fn mul(self, other : Self)
      ->Self{self * other} fn zero()
      ->Self{0} fn one()
      ->Self{1}
}

impl Numeric for i16 {
  fn add(self, other : Self) -> Self{self + other} fn mul(self, other : Self)
      ->Self{self * other} fn zero()
      ->Self{0} fn one()
      ->Self{1}
}

impl Numeric for u16 {
  fn add(self, other : Self) -> Self{self + other} fn mul(self, other : Self)
      ->Self{self * other} fn zero()
      ->Self{0} fn one()
      ->Self{1}
}

impl Numeric for i64 {
  fn add(self, other : Self) -> Self{self + other} fn mul(self, other : Self)
      ->Self{self * other} fn zero()
      ->Self{0} fn one()
      ->Self{1}
}

impl Numeric for u64 {
  fn add(self, other : Self) -> Self{self + other} fn mul(self, other : Self)
      ->Self{self * other} fn zero()
      ->Self{0} fn one()
      ->Self{1}
}

pub trait VectorOps<T : Numeric> : Sized {
  fn dot(self, other : Self)->T;
  fn cross(self, other : Self)->Self;
  fn magnitude(self)->T;
  fn normalize(self)->Option<Self>;
}

#[derive(Debug, Clone, Copy, Default)]
pub enum MathError {
#[default]
    DivisionByZero,
    InvalidInput,
    Overflow,
    Underflow,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum ShaderError {
    MathError(MathError),
    TextureError(&'static str),
    BufferError(&'static str),
#[default]
    InvalidState,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderState<T: Numeric> {
    pub transform: Matrix3x3<T>,
    pub position: Vec3<T>,
    pub color: Vec3<T>,
    pub material_id: Option<i32>,
    pub lighting_model: LightingModel,
}

impl<T: Numeric> RenderState<T> {
  pub fn new (transform : Matrix3x3<T>, position : Vec3<T>, color : Vec3<T>,
              material_id : Option<i32>, lighting_model_value : LightingModel)
      ->Self {
    Self {
      transform, position, color, material_id,
          lighting_model : lighting_model_value
    }
  }
}

#[derive(Debug, Clone, Copy)]
pub enum LightingModel {
    Phong { ambient: math::Vec3<f32>, diffuse: math::Vec3<f32>, specular: math::Vec3<f32>, shininess: f32 },
    PBR { albedo: math::Vec3<f32>, metallic: f32, roughness: f32, ao: f32 },
    Toon { base_color: math::Vec3<f32>, levels: i32, smoothing: f32 },
}

impl Default for LightingModel {
  fn default()->Self {
    Self::Phong {
    ambient:
      Default::default(), diffuse : Default::default(),
                                    specular : Default::default(),
                                               shininess : Default::default()
    }
  }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum VectorOp {
#[default]
  Add,
  Multiply,
  Cross,
  Normalize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Geometry {
  pub center : math::Vec3<f32>, pub normal : math::Vec3<f32>,
}

impl Geometry {
  pub fn new (center : math::Vec3<f32>, normal : math::Vec3<f32>)->Self {
    Self { center, normal }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Material {
  pub albedo : math::Vec3<f32>, pub roughness : f32,
}

impl Material {
  pub fn new (albedo : math::Vec3<f32>, roughness : f32)->Self {
    Self { albedo, roughness }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct LightingEnvironment {
  pub direction : math::Vec3<f32>, pub intensity : f32,
}

impl LightingEnvironment {
  pub fn new (direction : math::Vec3<f32>, intensity : f32)->Self {
    Self { direction, intensity }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderedFrame {
  pub color : Vec4<f32>, pub depth : f32,
}

impl RenderedFrame {
  pub fn new (color : Vec4<f32>, depth : f32)->Self {
    Self { color, depth }
  }
}

#[derive(Debug, Clone, Copy)]
pub enum RenderCommand {
    Draw { geometry: Geometry, material: Option<Material>, transform: Mat4<f32>, lighting: LightingEnvironment },
    Clear { color: Vec4<f32>, depth: f32 },
    SetState { state: RenderState<f32> },
}

impl Default for RenderCommand {
  fn default()->Self {
    Self::Draw {
    geometry:
      Default::default(), material : Default::default(),
                                     transform : Default::default(),
                                                 lighting : Default::default()
    }
  }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum RenderOutput {
  Success(RenderedFrame),
  Clear{color : Vec4<f32>, depth : f32},
#[default]
  StateSet,
}

pub fn safe_divide<T : Numeric + Copy + PartialEq + std::ops::Div<Output = T>>(
    a : T, b : T)
    ->Result<T, MathError> {
  match b {
    zero if (zero == T::zero()) =>{Result::Err(MathError::DivisionByZero)},
        _ =>{Result::Ok((a / b))},
  }
}

pub fn vector_operation<T : Numeric + Copy + std::ops::Sub<Output = T> +
                        std::ops::Add<Output = T> + CglSqrt + PartialEq +
                        std::ops::Div<Output = T>>(v1 : Vec3<T>, v2 : Vec3<T>,
                                                   op : VectorOp)
    -> Result<Vec3<T>, MathError> {
  match op {
    VectorOp::Add =>{Result::Ok(Vec3{
      x : Numeric::add(v1.x, v2.x),
      y : Numeric::add(v1.y, v2.y),
      z : Numeric::add(v1.z, v2.z)
    })},
        VectorOp::Multiply =>{Result::Ok(Vec3{
          x : Numeric::mul(v1.x, v2.x),
          y : Numeric::mul(v1.y, v2.y),
          z : Numeric::mul(v1.z, v2.z)
        })},
        VectorOp::Cross => {
      let cross_result : Vec3<T> = Vec3{
        x : (Numeric::mul(v1.y, v2.z) - Numeric::mul(v1.z, v2.y)),
        y : (Numeric::mul(v1.z, v2.x) - Numeric::mul(v1.x, v2.z)),
        z : (Numeric::mul(v1.x, v2.y) - Numeric::mul(v1.y, v2.x))
      };
      Result::Ok(cross_result)
    }
    , VectorOp::Normalize => {
      let mag_squared
          : T = ((Numeric::mul(v1.x, v1.x) + Numeric::mul(v1.y, v1.y)) +
                 Numeric::mul(v1.z, v1.z));
      match safe_divide(T::one(), CglSqrt::cgl_sqrt(mag_squared)) {
        Result::Ok(inv_mag) =>{Result::Ok(Vec3{
          x : Numeric::mul(v1.x, inv_mag),
          y : Numeric::mul(v1.y, inv_mag),
          z : Numeric::mul(v1.z, inv_mag)
        })},
            Result::Err(e) =>{Result::Err(e)},
      }
    }
    ,
  }
}

pub fn process_lighting_model(model : LightingModel,
                              light_dir : math::Vec3<f32>,
                              view_dir : math::Vec3<f32>,
                              normal : math::Vec3<f32>) -> math::Vec3<f32> {
  match model {
    LightingModel::Phong{ambient, diffuse, specular, shininess} => {
      let n_dot_l : f32 = max(0.0, dot(normal, light_dir));
      let reflect_dir : math::Vec3<f32> = reflect((-light_dir), normal);
      let r_dot_v : f32 = max(0.0, dot(reflect_dir, view_dir));
      ((ambient + math::Vec3::<f32>::new ((diffuse.x * n_dot_l),
                                          (diffuse.y * n_dot_l),
                                          (diffuse.z * n_dot_l))) +
       {
         let __cgl_vec_arg_0 = pow(r_dot_v, shininess);
         math::Vec3::<f32>::new ((specular.x * __cgl_vec_arg_0),
                                 (specular.y * __cgl_vec_arg_0),
                                 (specular.z * __cgl_vec_arg_0))
       })
    }
    , LightingModel::PBR{albedo, metallic, roughness, ao} => {
      let half_vector : math::Vec3<f32> = normalize((light_dir + view_dir));
      let n_dot_l : f32 = max(0.0, dot(normal, light_dir));
      let n_dot_v : f32 = max(0.0, dot(normal, view_dir));
      let n_dot_h : f32 = max(0.0, dot(normal, half_vector));
      let f0 : math::Vec3<f32> = lerp(math::Vec3::<f32>::new (0.04, 0.04, 0.04),
                                      albedo, metallic);
      let fresnel : math::Vec3<f32> = (f0 + {
        let __cgl_vec_arg_1 =
            math::Vec3::<f32>::new ((1.0 - f0.x), (1.0 - f0.y), (1.0 - f0.z));
        let __cgl_vec_arg_2 = pow((1.0 - n_dot_v), 5.0);
        math::Vec3::<f32>::new ((__cgl_vec_arg_1.x * __cgl_vec_arg_2),
                                (__cgl_vec_arg_1.y * __cgl_vec_arg_2),
                                (__cgl_vec_arg_1.z * __cgl_vec_arg_2))
      });
      let alpha : f32 = (roughness * roughness);
      let distribution
          : f32 = (alpha /
                   (3.14159 *
                    pow((((n_dot_h * n_dot_h) * (alpha - 1.0)) + 1.0), 2.0)));
      let geometry
          : f32 = ((n_dot_l * n_dot_v) /
                   (((n_dot_l + n_dot_v) - (n_dot_l * n_dot_v)) + 0.001));
      let brdf : math::Vec3<f32> = {
        let __cgl_vec_arg_4 = {let __cgl_vec_arg_3 = (distribution * geometry);
      math::Vec3::<f32>::new ((__cgl_vec_arg_3 * fresnel.x),
                              (__cgl_vec_arg_3 * fresnel.y),
                              (__cgl_vec_arg_3 * fresnel.z))
    };
    let __cgl_vec_arg_5 = (((4.0 * n_dot_l) * n_dot_v) + 0.001);
    math::Vec3::<f32>::new ((__cgl_vec_arg_4.x / __cgl_vec_arg_5),
                            (__cgl_vec_arg_4.y / __cgl_vec_arg_5),
                            (__cgl_vec_arg_4.z / __cgl_vec_arg_5))
  };
  let diffuse_contribution : math::Vec3<f32> = {
    let __cgl_vec_arg_8 = ({
      let __cgl_vec_arg_6 = math::Vec3::<f32>::new (
          (1.0 - fresnel.x), (1.0 - fresnel.y), (1.0 - fresnel.z));
      let __cgl_vec_arg_7 = (1.0 - metallic);
      math::Vec3::<f32>::new ((__cgl_vec_arg_6.x * __cgl_vec_arg_7),
                              (__cgl_vec_arg_6.y * __cgl_vec_arg_7),
                              (__cgl_vec_arg_6.z * __cgl_vec_arg_7))
    } * albedo);
  math::Vec3::<f32>::new ((__cgl_vec_arg_8.x / 3.14159),
                          (__cgl_vec_arg_8.y / 3.14159),
                          (__cgl_vec_arg_8.z / 3.14159))
};
{
  let __cgl_vec_arg_10 = { let __cgl_vec_arg_9 = (diffuse_contribution + brdf);
  math::Vec3::<f32>::new ((__cgl_vec_arg_9.x * n_dot_l),
                          (__cgl_vec_arg_9.y * n_dot_l),
                          (__cgl_vec_arg_9.z * n_dot_l))
};
math::Vec3::<f32>::new ((__cgl_vec_arg_10.x * ao), (__cgl_vec_arg_10.y * ao),
                        (__cgl_vec_arg_10.z * ao))
}
}
, LightingModel::Toon { base_color, levels, smoothing }
if (levels > 0)
  => {
    let n_dot_l : f32 = dot(normal, light_dir);
    let toon_level : f32 =
                         (floor((n_dot_l * (levels as f32))) / (levels as f32));
    let smooth_factor : f32 = smoothstep((toon_level - smoothing),
                                         (toon_level + smoothing), n_dot_l);
    math::Vec3::<f32>::new ((base_color.x * smooth_factor),
                            (base_color.y * smooth_factor),
                            (base_color.z * smooth_factor))
  }
,
    LightingModel::Toon{base_color, ..} =>{math::Vec3::<f32>::new (
        (base_color.x * 0.5), (base_color.y * 0.5), (base_color.z * 0.5))},
}
}

pub fn matrix_determinant<T : Numeric + Copy + std::ops::Sub<Output = T> +
                          std::ops::Add<Output = T>>(matrix : Matrix3x3<T>)
    -> T {
  match matrix {
    Matrix3x3{row0, row1, row2} => {
      let a : T = row0.x;
      let b : T = row0.y;
      let c : T = row0.z;
      let minor_a
          : T = (Numeric::mul(row1.y, row2.z) - Numeric::mul(row1.z, row2.y));
      let minor_b
          : T = (Numeric::mul(row1.x, row2.z) - Numeric::mul(row1.z, row2.x));
      let minor_c
          : T = (Numeric::mul(row1.x, row2.y) - Numeric::mul(row1.y, row2.x));
      ((Numeric::mul(a, minor_a) - Numeric::mul(b, minor_b)) +
       Numeric::mul(c, minor_c))
    }
    ,
  }
}

pub fn validate_material(material : Material) -> Result<i32, ShaderError> {
  match(material.roughness >= 0.0) {
    true =>{Result::Ok(1)}, false =>{Result::Err(ShaderError::InvalidState)},
  }
}

pub fn transform_geometry(shape : Geometry, transform : Mat4<f32>) -> Geometry {
  let transformed_center
      : Vec4<f32> =
            (transform * Vec4::<f32>::new (shape.center.x, shape.center.y,
                                           shape.center.z, 1.0));
  Geometry {
  center:
    math::Vec3::<f32>::new (transformed_center.x, transformed_center.y,
                            transformed_center.z),
        normal : shape.normal
  }
}

pub fn apply_lighting(shape : Geometry, material : Material,
                      lighting : LightingEnvironment)
    -> Result<RenderedFrame, MathError> {
  let light_strength
      : f32 = (max(0.0, dot(shape.normal, normalize(lighting.direction))) *
               lighting.intensity);
  let shaded_color
      : math::Vec3<f32> =
            math::Vec3::<f32>::new ((material.albedo.x * light_strength),
                                    (material.albedo.y * light_strength),
                                    (material.albedo.z * light_strength));
  Result::Ok(RenderedFrame{
    color :
        Vec4::<f32>::new (shaded_color.x, shaded_color.y, shaded_color.z, 1.0),
    depth : 1.0
  })
}

pub fn validate_state(state : RenderState<f32>) -> bool {
  match state.material_id {
    Option::Some(id) =>{(id >= 0)}, Option::None =>{true},
  }
}

pub fn process_render_command(command : RenderCommand)
    -> Result<RenderOutput, ShaderError> {
  match command {
    RenderCommand::
    Draw{geometry, material : Option::Some(material), transform, lighting} =>{
        match validate_material(material){
            Result::Ok(_) =>{let transformed_geometry : Geometry =
                                 transform_geometry(geometry, transform);
    let lit_result
        : Result<RenderedFrame, MathError> =
              apply_lighting(transformed_geometry, material, lighting);
    match lit_result {
      Result::Ok(output) =>{Result::Ok(RenderOutput::Success(output))},
          Result::Err(lighting_error) =>{
              Result::Err(ShaderError::MathError(lighting_error))},
    }
  }
  , Result::Err(_validation_error) =>{Result::Err(ShaderError::InvalidState)},
}
}
, RenderCommand::Draw{geometry : _geometry,
                      material : Option::None,
                      ..} =>{Result::Err(ShaderError::InvalidState)},
    RenderCommand::Clear{color, depth} =>{
        Result::Ok(RenderOutput::Clear{color : color, depth : depth})},
    RenderCommand::SetState{state} =>{match validate_state(state){
        true =>{Result::Ok(RenderOutput::StateSet)},
        false =>{Result::Err(ShaderError::InvalidState)},
    }},
}
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VertexInput {
  pub position : math::Vec3<f32>, pub normal : math::Vec3<f32>,
      pub uv : Vec2<f32>, pub color : Vec4<f32>,
}

impl VertexInput {
  pub fn new (position : math::Vec3<f32>, normal : math::Vec3<f32>,
              uv : Vec2<f32>, color : Vec4<f32>)
      ->Self {
    Self { position, normal, uv, color }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VertexOutput {
  pub position : Vec4<f32>, pub world_position : math::Vec3<f32>,
      pub normal : math::Vec3<f32>, pub uv : Vec2<f32>, pub color : Vec4<f32>,
}

impl VertexOutput {
  pub fn new (position : Vec4<f32>, world_position : math::Vec3<f32>,
              normal : math::Vec3<f32>, uv : Vec2<f32>, color : Vec4<f32>)
      ->Self {
    Self { position, world_position, normal, uv, color }
  }
}

static MVP_MATRIX : std::sync::LazyLock<Mat4<f32>> =
                        std::sync::LazyLock::new (|| Default::default());
static MODEL_MATRIX : std::sync::LazyLock<Mat4<f32>> =
                          std::sync::LazyLock::new (|| Default::default());
static NORMAL_MATRIX : std::sync::LazyLock<Mat3<f32>> =
                           std::sync::LazyLock::new (|| Default::default());
// Vertex Shader
#[cfg_attr(feature = "crossgl_gpu", vertex_shader)]
pub fn vertex_main(input : VertexInput) -> VertexOutput {
  let world_pos
      : Vec4<f32> = (*MODEL_MATRIX * Vec4::<f32>::new (input.position.x,
                                                       input.position.y,
                                                       input.position.z, 1.0));
  let clip_pos
      : Vec4<f32> =
            (*MVP_MATRIX * Vec4::<f32>::new (input.position.x, input.position.y,
                                             input.position.z, 1.0));
  let world_normal : math::Vec3<f32> =
                         normalize((*NORMAL_MATRIX * input.normal));
  let normal_vec3
      : Vec3<f32> =
        Vec3{x : world_normal.x, y : world_normal.y, z : world_normal.z};
  let processed_normal
      : math::Vec3<f32> =
            match vector_operation(normal_vec3, normal_vec3,
                                   VectorOp::Normalize){
                Result::Ok(normalized) =>{math::Vec3::<f32>::new (
                    normalized.x, normalized.y, normalized.z)},
                Result::Err(_) =>{world_normal},
      };
  VertexOutput {
  position:
    clip_pos,
        world_position
        : math::Vec3::<f32>::new (world_pos.x, world_pos.y, world_pos.z),
          normal : processed_normal,
                   uv : input.uv,
                   color : input.color
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FragmentInput {
  pub world_position : math::Vec3<f32>, pub normal : math::Vec3<f32>,
      pub uv : Vec2<f32>, pub color : Vec4<f32>,
}

impl FragmentInput {
  pub fn new (world_position : math::Vec3<f32>, normal : math::Vec3<f32>,
              uv : Vec2<f32>, color : Vec4<f32>)
      ->Self {
    Self { world_position, normal, uv, color }
  }
}

static LIGHTING_MODEL : std::sync::LazyLock<LightingModel> =
                            std::sync::LazyLock::new (|| Default::default());
static LIGHT_DIRECTION : std::sync::LazyLock<math::Vec3<f32>> =
                             std::sync::LazyLock::new (|| Default::default());
static CAMERA_POSITION : std::sync::LazyLock<math::Vec3<f32>> =
                             std::sync::LazyLock::new (|| Default::default());
// CrossGL resource metadata: name=main_texture kind=texture set=0 binding=0
// binding_source=automatic CrossGL Rust limitation: resource main_texture is
// emitted as a compile-only placeholder static, not a rust-gpu resource
// binding; pass real spirv_std resources as #[spirv(...)] entry parameters when
// targeting rust-gpu.
static MAIN_TEXTURE : std::sync::LazyLock<Texture2D<f32>> =
                          std::sync::LazyLock::new (|| Default::default());
// Fragment Shader
#[cfg_attr(feature = "crossgl_gpu", fragment_shader)]
pub fn fragment_main(input : FragmentInput) -> Vec4<f32> {
  let view_dir : math::Vec3<f32> =
                     normalize((*CAMERA_POSITION - input.world_position));
  let normal : math::Vec3<f32> = normalize(input.normal);
  let light_dir : math::Vec3<f32> = normalize((-*LIGHT_DIRECTION));
  let tex_color : Vec4<f32> = sample(*MAIN_TEXTURE, input.uv);
  let lighting_contribution : math::Vec3<f32> = process_lighting_model(
                                  *LIGHTING_MODEL, light_dir, view_dir, normal);
  let final_color
      : math::Vec3<f32> =
            ((math::Vec3::<f32>::new (tex_color.x, tex_color.y, tex_color.z) *
              lighting_contribution) *
             math::Vec3::<f32>::new (input.color.x, input.color.y,
                                     input.color.z));
  Vec4::<f32>::new (final_color.x, final_color.y, final_color.z,
                    (tex_color.w * input.color.w))
}

pub trait CglSqrt { fn cgl_sqrt(self)->Self; }

impl CglSqrt for f32 {
  fn cgl_sqrt(self) -> Self { self.sqrt() }
}

impl CglSqrt for f64 {
  fn cgl_sqrt(self) -> Self { self.sqrt() }
}

impl CglSqrt for i32 {
  fn cgl_sqrt(self) -> Self { (self as f64).sqrt() as i32 }
}

impl CglSqrt for u32 {
  fn cgl_sqrt(self) -> Self { (self as f64).sqrt() as u32 }
}

impl CglSqrt for i16 {
  fn cgl_sqrt(self) -> Self { (self as f64).sqrt() as i16 }
}

impl CglSqrt for u16 {
  fn cgl_sqrt(self) -> Self { (self as f64).sqrt() as u16 }
}

impl CglSqrt for i64 {
  fn cgl_sqrt(self) -> Self { (self as f64).sqrt() as i64 }
}

impl CglSqrt for u64 {
  fn cgl_sqrt(self) -> Self { (self as f64).sqrt() as u64 }
}
