// Generated Rust GPU Shader Code
use gpu::*;
use math::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MaterialProperties {
  pub albedo : Vec3<f32>, pub metallic : f32, pub roughness : f32, pub ao : f32,
      pub emission : Vec3<f32>, pub normal_scale : f32, pub height_scale : f32,
      pub has_albedo_map : bool, pub has_normal_map : bool,
      pub has_metallic_roughness_map : bool, pub has_ao_map : bool,
      pub has_emission_map : bool, pub has_height_map : bool,
}

impl MaterialProperties {
  pub fn new (albedo : Vec3<f32>, metallic : f32, roughness : f32, ao : f32,
              emission : Vec3<f32>, normal_scale : f32, height_scale : f32,
              has_albedo_map : bool, has_normal_map : bool,
              has_metallic_roughness_map : bool, has_ao_map : bool,
              has_emission_map : bool, has_height_map : bool)
      ->Self {
    Self {
      albedo, metallic, roughness, ao, emission, normal_scale, height_scale,
          has_albedo_map, has_normal_map, has_metallic_roughness_map,
          has_ao_map, has_emission_map, has_height_map
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct LightData {
  pub position : Vec3<f32>, pub direction : Vec3<f32>, pub color : Vec3<f32>,
      pub intensity : f32, pub range : f32, pub inner_cone_angle : f32,
      pub outer_cone_angle : f32, pub type_ : i32, pub cast_shadows : bool,
      pub light_view_proj : Mat4<f32>,
}

impl LightData {
  pub fn new (position : Vec3<f32>, direction : Vec3<f32>, color : Vec3<f32>,
              intensity : f32, range : f32, inner_cone_angle : f32,
              outer_cone_angle : f32, type_ : i32, cast_shadows : bool,
              light_view_proj : Mat4<f32>)
      ->Self {
    Self {
      position, direction, color, intensity, range, inner_cone_angle,
          outer_cone_angle, type_, cast_shadows, light_view_proj
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EnvironmentData {
  pub irradiance_map : TextureCube<f32>, pub prefilter_map : TextureCube<f32>,
      pub brdf_lut : Texture2D<f32>, pub max_reflection_lod : f32,
      pub exposure : f32, pub ambient_color : Vec3<f32>,
}

impl EnvironmentData {
  pub fn new (irradiance_map_value : TextureCube<f32>,
              prefilter_map : TextureCube<f32>, brdf_lut : Texture2D<f32>,
              max_reflection_lod : f32, exposure : f32,
              ambient_color : Vec3<f32>)
      ->Self {
    Self {
    irradiance_map:
      irradiance_map_value, prefilter_map, brdf_lut, max_reflection_lod,
          exposure, ambient_color
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CameraData {
  pub position : Vec3<f32>, pub forward : Vec3<f32>, pub up : Vec3<f32>,
      pub right : Vec3<f32>, pub view_matrix : Mat4<f32>,
      pub projection_matrix : Mat4<f32>, pub view_projection_matrix : Mat4<f32>,
      pub near_plane : f32, pub far_plane : f32, pub fov : f32,
      pub screen_size : Vec2<f32>,
}

impl CameraData {
  pub fn new (position : Vec3<f32>, forward : Vec3<f32>, up : Vec3<f32>,
              right : Vec3<f32>, view_matrix_value : Mat4<f32>,
              projection_matrix_value : Mat4<f32>,
              view_projection_matrix : Mat4<f32>, near_plane : f32,
              far_plane : f32, fov : f32, screen_size : Vec2<f32>)
      ->Self {
    Self {
      position, forward, up, right,
          view_matrix : view_matrix_value,
                        projection_matrix : projection_matrix_value,
                                            view_projection_matrix,
                                            near_plane,
                                            far_plane,
                                            fov,
                                            screen_size
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderSettings {
  pub enable_ibl : bool, pub enable_shadows : bool,
      pub enable_normal_mapping : bool, pub enable_parallax_mapping : bool,
      pub enable_tone_mapping : bool, pub enable_gamma_correction : bool,
      pub shadow_cascade_count : i32, pub shadow_bias : f32,
      pub max_lights : i32, pub lod_bias : f32,
}

impl RenderSettings {
  pub fn new (enable_ibl : bool, enable_shadows : bool,
              enable_normal_mapping : bool, enable_parallax_mapping : bool,
              enable_tone_mapping : bool, enable_gamma_correction : bool,
              shadow_cascade_count : i32, shadow_bias : f32, max_lights : i32,
              lod_bias : f32)
      ->Self {
    Self {
      enable_ibl, enable_shadows, enable_normal_mapping,
          enable_parallax_mapping, enable_tone_mapping, enable_gamma_correction,
          shadow_cascade_count, shadow_bias, max_lights, lod_bias
    }
  }
}

const PI : f32 = 3.14159265359;
const EPSILON : f32 = 0.0001;
const MAX_LIGHTS : i32 = 32;
const MAX_SHADOW_CASCADES : i32 = 4;

// Constant Buffers
pub fn getNormalFromMap(normal_map : Texture2D<f32>, uv : Vec2<f32>,
                        tbn : Mat3<f32>, scale : f32) -> Vec3<f32> {
  let mut tangent_normal : Vec3<f32> = {
    let __cgl_vec_arg_1 = {let __cgl_vec_arg_0 = sample(normal_map, uv);
  Vec3::<f32>::new ((__cgl_vec_arg_0.x * 2.0), (__cgl_vec_arg_0.y * 2.0),
                    (__cgl_vec_arg_0.z * 2.0))
};
Vec3::<f32>::new ((__cgl_vec_arg_1.x - 1.0), (__cgl_vec_arg_1.y - 1.0),
                  (__cgl_vec_arg_1.z - 1.0))
}
;
{
  let __cgl_swizzle_0 = scale;
  tangent_normal.x *= __cgl_swizzle_0;
  tangent_normal.y *= __cgl_swizzle_0;
};
return normalize((tbn * tangent_normal));
}

pub fn parallaxMapping(height_map : Texture2D<f32>, uv : Vec2<f32>,
                       view_dir : Vec3<f32>, height_scale : f32) -> Vec2<f32> {
  let height : f32 = sample(height_map, uv).x;
  let p : Vec2<f32> = {
    let __cgl_vec_arg_2 =
        Vec2::<f32>::new ((view_dir.x / view_dir.z), (view_dir.y / view_dir.z));
  let __cgl_vec_arg_3 = (height * height_scale);
  Vec2::<f32>::new ((__cgl_vec_arg_2.x * __cgl_vec_arg_3),
                    (__cgl_vec_arg_2.y * __cgl_vec_arg_3))
};
return Vec2::<f32>::new ((uv.x - p.x), (uv.y - p.y));
}

pub fn distributionGGX(N : Vec3<f32>, H : Vec3<f32>, roughness : f32) -> f32 {
  let a : f32 = (roughness * roughness);
  let a2 : f32 = (a * a);
  let NdotH : f32 = max(dot(N, H), 0.0);
  let NdotH2 : f32 = (NdotH * NdotH);
  let num : f32 = a2;
  let mut denom : f32 = ((NdotH2 * (a2 - 1.0)) + 1.0);
  denom = ((PI * denom) * denom);
  return (num / max(denom, EPSILON));
}

pub fn geometrySchlickGGX(NdotV : f32, roughness : f32) -> f32 {
  let r : f32 = (roughness + 1.0);
  let k : f32 = ((r * r) / 8.0);
  let num : f32 = NdotV;
  let denom : f32 = ((NdotV * (1.0 - k)) + k);
  return (num / max(denom, EPSILON));
}

pub fn geometrySmith(N : Vec3<f32>, V : Vec3<f32>, L : Vec3<f32>,
                     roughness : f32) -> f32 {
  let NdotV : f32 = max(dot(N, V), 0.0);
  let NdotL : f32 = max(dot(N, L), 0.0);
  let ggx2 : f32 = geometrySchlickGGX(NdotV, roughness);
  let ggx1 : f32 = geometrySchlickGGX(NdotL, roughness);
  return (ggx1 * ggx2);
}

pub fn fresnelSchlick(cosTheta : f32, F0 : Vec3<f32>) -> Vec3<f32> {
  return (F0 + {
    let __cgl_vec_arg_4 =
        Vec3::<f32>::new ((1.0 - F0.x), (1.0 - F0.y), (1.0 - F0.z));
    let __cgl_vec_arg_5 = pow(max((1.0 - cosTheta), 0.0), 5.0);
    Vec3::<f32>::new ((__cgl_vec_arg_4.x * __cgl_vec_arg_5),
                      (__cgl_vec_arg_4.y * __cgl_vec_arg_5),
                      (__cgl_vec_arg_4.z * __cgl_vec_arg_5))
  });
}

pub fn fresnelSchlickRoughness(cosTheta : f32, F0 : Vec3<f32>, roughness : f32)
    -> Vec3<f32> {
  return (F0 + {
    let __cgl_vec_arg_7 =
        (max(
             {
               let __cgl_vec_arg_6 = (1.0 - roughness);
               Vec3::<f32>::new (__cgl_vec_arg_6, __cgl_vec_arg_6,
                                 __cgl_vec_arg_6)
             },
             F0) -
         F0);
    let __cgl_vec_arg_8 = pow(max((1.0 - cosTheta), 0.0), 5.0);
    Vec3::<f32>::new ((__cgl_vec_arg_7.x * __cgl_vec_arg_8),
                      (__cgl_vec_arg_7.y * __cgl_vec_arg_8),
                      (__cgl_vec_arg_7.z * __cgl_vec_arg_8))
  });
}

pub fn calculateShadow(shadow_map : Texture2D<f32>,
                       frag_pos_light_space : Vec4<f32>, bias : f32) -> f32 {
  let mut proj_coords : Vec3<f32> = Vec3::<f32>::new (
                            (frag_pos_light_space.x / frag_pos_light_space.w),
                            (frag_pos_light_space.y / frag_pos_light_space.w),
                            (frag_pos_light_space.z / frag_pos_light_space.w));
  proj_coords = {
    let __cgl_vec_arg_9 = Vec3::<f32>::new (
        (proj_coords.x * 0.5), (proj_coords.y * 0.5), (proj_coords.z * 0.5));
  Vec3::<f32>::new ((__cgl_vec_arg_9.x + 0.5), (__cgl_vec_arg_9.y + 0.5),
                    (__cgl_vec_arg_9.z + 0.5))
};
if (proj_coords.z > 1.0) {
}
let mut shadow : f32 = 0.0;
let texel_size : Vec2<f32> = {
  let __cgl_vec_arg_10 : Vec2<i32> = texture_size_lod(shadow_map, 0);
Vec2::<f32>::new ((1.0 / (__cgl_vec_arg_10.x as f32)),
                  (1.0 / (__cgl_vec_arg_10.y as f32)))
}
;
let mut x : i32 = (-1);
while (x <= 1) {
  let mut y : i32 = (-1);
  while (y <= 1) {
            let pcf_depth: f32 = sample(shadow_map, {
      let __cgl_vec_arg_12 = {
        let __cgl_vec_arg_11 = Vec2::<f32>::new ((x as f32), (y as f32));
      Vec2::<f32>::new ((__cgl_vec_arg_11.x * texel_size.x),
                        (__cgl_vec_arg_11.y * texel_size.y)) }; Vec2::<f32>::new((proj_coords.x + __cgl_vec_arg_12.x), (proj_coords.y + __cgl_vec_arg_12.y))
  }).x;
  shadow += (if ((proj_coords.z - bias) > pcf_depth){1.0} else {0.0});
  y += 1;
}
x += 1;
}
return (shadow / 9.0);
}

pub fn calculateIBL(N : Vec3<f32>, V : Vec3<f32>, albedo : Vec3<f32>,
                    metallic : f32, roughness : f32, env : EnvironmentData)
    -> Vec3<f32> {
  let F0 : Vec3<f32> =
               lerp(Vec3::<f32>::new (0.04, 0.04, 0.04), albedo, metallic);
  let F : Vec3<f32> =
              fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
  let kS : Vec3<f32> = F;
  let mut kD : Vec3<f32> =
                   Vec3::<f32>::new ((1.0 - kS.x), (1.0 - kS.y), (1.0 - kS.z));
  kD = { let __cgl_vec_arg_13 = (1.0 - metallic);
  Vec3::<f32>::new ((kD.x * __cgl_vec_arg_13), (kD.y * __cgl_vec_arg_13),
                    (kD.z * __cgl_vec_arg_13))
};
let irradiance : Vec3<f32> = {
  let __cgl_swizzle_1 = sample(env.irradiance_map, N);
Vec3::<f32>::new (__cgl_swizzle_1.x, __cgl_swizzle_1.y, __cgl_swizzle_1.z)
}
;
let diffuse : Vec3<f32> = (irradiance * albedo);
let R : Vec3<f32> = reflect((-V), N);
let prefiltered_color : Vec3<f32> = {
  let __cgl_swizzle_2 =
      sample_lod(env.prefilter_map, R, (roughness * env.max_reflection_lod));
Vec3::<f32>::new (__cgl_swizzle_2.x, __cgl_swizzle_2.y, __cgl_swizzle_2.z)
}
;
let brdf : Vec2<f32> = {
  let __cgl_swizzle_3 =
      sample(env.brdf_lut, Vec2::<f32>::new (max(dot(N, V), 0.0), roughness));
Vec2::<f32>::new (__cgl_swizzle_3.x, __cgl_swizzle_3.y)
}
;
let specular : Vec3<f32> = (prefiltered_color * {
  let __cgl_vec_arg_14 =
      Vec3::<f32>::new ((F.x * brdf.x), (F.y * brdf.x), (F.z * brdf.x));
  Vec3::<f32>::new ((__cgl_vec_arg_14.x + brdf.y),
                    (__cgl_vec_arg_14.y + brdf.y),
                    (__cgl_vec_arg_14.z + brdf.y))
});
return {
  let __cgl_vec_arg_15 = ((kD * diffuse) + specular);
  Vec3::<f32>::new ((__cgl_vec_arg_15.x * env.exposure),
                    (__cgl_vec_arg_15.y * env.exposure),
                    (__cgl_vec_arg_15.z * env.exposure))
};
}

pub fn calculateDirectLighting(N : Vec3<f32>, V : Vec3<f32>, L : Vec3<f32>,
                               albedo : Vec3<f32>, metallic : f32,
                               roughness : f32, light_color : Vec3<f32>)
    -> Vec3<f32> {
  let H : Vec3<f32> = normalize((V + L));
  let F0 : Vec3<f32> =
               lerp(Vec3::<f32>::new (0.04, 0.04, 0.04), albedo, metallic);
  let NDF : f32 = distributionGGX(N, H, roughness);
  let G : f32 = geometrySmith(N, V, L, roughness);
  let F : Vec3<f32> = fresnelSchlick(max(dot(H, V), 0.0), F0);
  let kS : Vec3<f32> = F;
  let mut kD : Vec3<f32> = (Vec3::<f32>::new (1.0, 1.0, 1.0) - kS);
  kD = { let __cgl_vec_arg_16 = (1.0 - metallic);
  Vec3::<f32>::new ((kD.x * __cgl_vec_arg_16), (kD.y * __cgl_vec_arg_16),
                    (kD.z * __cgl_vec_arg_16))
};
let numerator : Vec3<f32> = { let __cgl_vec_arg_17 = (NDF * G);
Vec3::<f32>::new ((__cgl_vec_arg_17 * F.x), (__cgl_vec_arg_17 * F.y),
                  (__cgl_vec_arg_17 * F.z))
}
;
let denominator
    : f32 = (((4.0 * max(dot(N, V), 0.0)) * max(dot(N, L), 0.0)) + EPSILON);
let specular : Vec3<f32> = Vec3::<f32>::new ((numerator.x / denominator),
                                             (numerator.y / denominator),
                                             (numerator.z / denominator));
let NdotL : f32 = max(dot(N, L), 0.0);
return {
  let __cgl_vec_arg_19 =
      (({
         let __cgl_vec_arg_18 = (kD * albedo);
         Vec3::<f32>::new ((__cgl_vec_arg_18.x / PI), (__cgl_vec_arg_18.y / PI),
                           (__cgl_vec_arg_18.z / PI))
       } +
        specular) *
       light_color);
  Vec3::<f32>::new ((__cgl_vec_arg_19.x * NdotL), (__cgl_vec_arg_19.y * NdotL),
                    (__cgl_vec_arg_19.z * NdotL))
};
}

pub fn calculateAttenuation(light : LightData, frag_pos : Vec3<f32>) -> f32 {
  if (light.type_ == 0) {
    return 1.0;
  }
  let distance : f32 = length((light.position - frag_pos));
  if (light.type_ == 1) {
    let attenuation : f32 = (1.0 / (distance * distance));
    return (attenuation * smoothstep(light.range, 0.0, distance));
  }
  if (light.type_ == 2) {
    let light_dir : Vec3<f32> = normalize((light.position - frag_pos));
    let theta : f32 = dot(light_dir, normalize((-light.direction)));
    let epsilon : f32 = (light.inner_cone_angle - light.outer_cone_angle);
    let intensity
        : f32 = clamp(((theta - light.outer_cone_angle) / epsilon), 0.0, 1.0);
    let attenuation : f32 = (1.0 / (distance * distance));
    return ((attenuation * intensity) * smoothstep(light.range, 0.0, distance));
  }
  return 0.0;
}

pub fn reinhardToneMapping(color : Vec3<f32>) -> Vec3<f32> {
  return {
    let __cgl_vec_arg_20 = (color + Vec3::<f32>::new (1.0, 1.0, 1.0));
    Vec3::<f32>::new ((color.x / __cgl_vec_arg_20.x),
                      (color.y / __cgl_vec_arg_20.y),
                      (color.z / __cgl_vec_arg_20.z))
  };
}

pub fn acesToneMapping(color : Vec3<f32>) -> Vec3<f32> {
  let a : f32 = 2.51;
  let b : f32 = 0.03;
  let c : f32 = 2.43;
  let d : f32 = 0.59;
  let e : f32 = 0.14;
    return clamp({
    let __cgl_vec_arg_22 = (color * {
      let __cgl_vec_arg_21 =
          Vec3::<f32>::new ((a * color.x), (a * color.y), (a * color.z));
      Vec3::<f32>::new ((__cgl_vec_arg_21.x + b), (__cgl_vec_arg_21.y + b),
                        (__cgl_vec_arg_21.z + b))
    });
    let __cgl_vec_arg_25 = {
      let __cgl_vec_arg_24 = (color * {
        let __cgl_vec_arg_23 =
            Vec3::<f32>::new ((c * color.x), (c * color.y), (c * color.z));
        Vec3::<f32>::new ((__cgl_vec_arg_23.x + d), (__cgl_vec_arg_23.y + d),
                          (__cgl_vec_arg_23.z + d))
      });
    Vec3::<f32>::new ((__cgl_vec_arg_24.x + e), (__cgl_vec_arg_24.y + e),
                      (__cgl_vec_arg_24.z + e)) }; Vec3::<f32>::new((__cgl_vec_arg_22.x / __cgl_vec_arg_25.x), (__cgl_vec_arg_22.y / __cgl_vec_arg_25.y), (__cgl_vec_arg_22.z / __cgl_vec_arg_25.z))
}, 0.0, 1.0);
}

pub fn gammaCorrection(color : Vec3<f32>, gamma : f32) -> Vec3<f32> {
  return pow(color, {
    let __cgl_vec_arg_26 = (1.0 / gamma);
    Vec3::<f32>::new (__cgl_vec_arg_26, __cgl_vec_arg_26, __cgl_vec_arg_26)
  });
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VertexInput {
  pub position : Vec3<f32>, pub normal : Vec3<f32>, pub tangent : Vec3<f32>,
      pub uv : Vec2<f32>, pub color : Vec4<f32>,
}

impl VertexInput {
  pub fn new (position : Vec3<f32>, normal : Vec3<f32>, tangent : Vec3<f32>,
              uv : Vec2<f32>, color : Vec4<f32>)
      ->Self {
    Self { position, normal, tangent, uv, color }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct VertexOutput {
  pub clip_position : Vec4<f32>, pub world_position : Vec3<f32>,
      pub world_normal : Vec3<f32>, pub world_tangent : Vec3<f32>,
      pub world_bitangent : Vec3<f32>, pub uv : Vec2<f32>,
      pub color : Vec4<f32>, pub tbn_matrix : Mat3<f32>,
      pub shadow_coords : Vec<Vec4<f32>>,
}

impl VertexOutput {
  pub fn new (clip_position : Vec4<f32>, world_position : Vec3<f32>,
              world_normal : Vec3<f32>, world_tangent : Vec3<f32>,
              world_bitangent : Vec3<f32>, uv : Vec2<f32>, color : Vec4<f32>,
              tbn_matrix : Mat3<f32>, shadow_coords : Vec<Vec4<f32>>)
      ->Self {
    Self {
      clip_position, world_position, world_normal, world_tangent,
          world_bitangent, uv, color, tbn_matrix, shadow_coords
    }
  }
}

static MODEL_MATRIX : std::sync::LazyLock<Mat4<f32>> =
                          std::sync::LazyLock::new (|| Default::default());
static VIEW_MATRIX : std::sync::LazyLock<Mat4<f32>> =
                         std::sync::LazyLock::new (|| Default::default());
static PROJECTION_MATRIX : std::sync::LazyLock<Mat4<f32>> =
                               std::sync::LazyLock::new (|| Default::default());
static NORMAL_MATRIX : std::sync::LazyLock<Mat3<f32>> =
                           std::sync::LazyLock::new (|| Default::default());
static CAMERA : std::sync::LazyLock<CameraData> =
                    std::sync::LazyLock::new (|| Default::default());
static SETTINGS : std::sync::LazyLock<RenderSettings> =
                      std::sync::LazyLock::new (|| Default::default());
static SHADOW_MATRICES : Vec<Mat4<f32>> = Vec::new ();
// Vertex Shader
#[cfg_attr(feature = "crossgl_gpu", vertex_shader)]
pub fn vertex_main(input : VertexInput) -> VertexOutput {
  let mut output : VertexOutput = Default::default();
  let world_pos
      : Vec4<f32> = (*MODEL_MATRIX * Vec4::<f32>::new (input.position.x,
                                                       input.position.y,
                                                       input.position.z, 1.0));
  output.world_position =
      Vec3::<f32>::new (world_pos.x, world_pos.y, world_pos.z);
  output.clip_position = ((*CAMERA).view_projection_matrix * world_pos);
  output.world_normal = normalize((*NORMAL_MATRIX * input.normal));
  output.world_tangent = normalize((*NORMAL_MATRIX * input.tangent));
  output.world_bitangent = cross(output.world_normal, output.world_tangent);
  output.tbn_matrix = Mat3::<f32>::new (
      output.world_tangent.x, output.world_tangent.y, output.world_tangent.z,
      output.world_bitangent.x, output.world_bitangent.y,
      output.world_bitangent.z, output.world_normal.x, output.world_normal.y,
      output.world_normal.z);
  output.uv = input.uv;
  output.color = input.color;
  if (*SETTINGS)
    .enable_shadows {
      let mut i : i32 = 0;
      while ((i < (*SETTINGS).shadow_cascade_count) &&
             (i < MAX_SHADOW_CASCADES)) {
        output.shadow_coords[i as usize] =
            (SHADOW_MATRICES[i as usize] * world_pos);
        i += 1;
      }
    }
  return output;
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct FragmentInput {
  pub world_position : Vec3<f32>, pub world_normal : Vec3<f32>,
      pub world_tangent : Vec3<f32>, pub world_bitangent : Vec3<f32>,
      pub uv : Vec2<f32>, pub color : Vec4<f32>, pub tbn_matrix : Mat3<f32>,
      pub shadow_coords : Vec<Vec4<f32>>,
}

impl FragmentInput {
  pub fn new (world_position : Vec3<f32>, world_normal : Vec3<f32>,
              world_tangent : Vec3<f32>, world_bitangent : Vec3<f32>,
              uv : Vec2<f32>, color : Vec4<f32>, tbn_matrix : Mat3<f32>,
              shadow_coords : Vec<Vec4<f32>>)
      ->Self {
    Self {
      world_position, world_normal, world_tangent, world_bitangent, uv, color,
          tbn_matrix, shadow_coords
    }
  }
}

// CrossGL resource metadata: name=albedo_map kind=texture set=0 binding=0
// binding_source=automatic CrossGL Rust limitation: resource albedo_map is
// emitted as a compile-only placeholder static, not a rust-gpu resource
// binding; pass real spirv_std resources as #[spirv(...)] entry parameters when
// targeting rust-gpu.
static ALBEDO_MAP : std::sync::LazyLock<Texture2D<f32>> =
                        std::sync::LazyLock::new (|| Default::default());
// CrossGL resource metadata: name=normal_map kind=texture set=0 binding=1
// binding_source=automatic CrossGL Rust limitation: resource normal_map is
// emitted as a compile-only placeholder static, not a rust-gpu resource
// binding; pass real spirv_std resources as #[spirv(...)] entry parameters when
// targeting rust-gpu.
static NORMAL_MAP : std::sync::LazyLock<Texture2D<f32>> =
                        std::sync::LazyLock::new (|| Default::default());
// CrossGL resource metadata: name=metallic_roughness_map kind=texture set=0
// binding=2 binding_source=automatic CrossGL Rust limitation: resource
// metallic_roughness_map is emitted as a compile-only placeholder static, not a
// rust-gpu resource binding; pass real spirv_std resources as #[spirv(...)]
// entry parameters when targeting rust-gpu.
static METALLIC_ROUGHNESS_MAP
    : std::sync::LazyLock<Texture2D<f32>> =
          std::sync::LazyLock::new (|| Default::default());
// CrossGL resource metadata: name=ao_map kind=texture set=0 binding=3
// binding_source=automatic CrossGL Rust limitation: resource ao_map is emitted
// as a compile-only placeholder static, not a rust-gpu resource binding; pass
// real spirv_std resources as #[spirv(...)] entry parameters when targeting
// rust-gpu.
static AO_MAP : std::sync::LazyLock<Texture2D<f32>> =
                    std::sync::LazyLock::new (|| Default::default());
// CrossGL resource metadata: name=emission_map kind=texture set=0 binding=4
// binding_source=automatic CrossGL Rust limitation: resource emission_map is
// emitted as a compile-only placeholder static, not a rust-gpu resource
// binding; pass real spirv_std resources as #[spirv(...)] entry parameters when
// targeting rust-gpu.
static EMISSION_MAP : std::sync::LazyLock<Texture2D<f32>> =
                          std::sync::LazyLock::new (|| Default::default());
// CrossGL resource metadata: name=height_map kind=texture set=0 binding=5
// binding_source=automatic CrossGL Rust limitation: resource height_map is
// emitted as a compile-only placeholder static, not a rust-gpu resource
// binding; pass real spirv_std resources as #[spirv(...)] entry parameters when
// targeting rust-gpu.
static HEIGHT_MAP : std::sync::LazyLock<Texture2D<f32>> =
                        std::sync::LazyLock::new (|| Default::default());
// CrossGL resource metadata: name=shadow_maps kind=texture set=0 binding=6
// binding_source=automatic CrossGL Rust limitation: resource shadow_maps is
// emitted as a compile-only placeholder static, not a rust-gpu resource
// binding; pass real spirv_std resources as #[spirv(...)] entry parameters when
// targeting rust-gpu.
static SHADOW_MAPS : Vec<Texture2D<f32>> = Vec::new ();
static MATERIAL : std::sync::LazyLock<MaterialProperties> =
                      std::sync::LazyLock::new (|| Default::default());
static ENVIRONMENT : std::sync::LazyLock<EnvironmentData> =
                         std::sync::LazyLock::new (|| Default::default());
static LIGHTS : Vec<LightData> = Vec::new ();
static ACTIVE_LIGHT_COUNT : i32 = 0;
// Fragment Shader
#[cfg_attr(feature = "crossgl_gpu", fragment_shader)]
pub fn fragment_main(input : FragmentInput) -> Vec4<f32> {
  let mut uv : Vec2<f32> = input.uv;
  if ((*SETTINGS).enable_parallax_mapping && (*MATERIAL).has_height_map) {
    let view_dir : Vec3<f32> =
                       normalize(((*CAMERA).position - input.world_position));
    let tangent_view_dir : Vec3<f32> = (transpose(input.tbn_matrix) * view_dir);
    uv = parallaxMapping(*HEIGHT_MAP, uv, tangent_view_dir,
                         (*MATERIAL).height_scale);
  }
  let mut albedo : Vec3<f32> = (*MATERIAL).albedo;
  if (*MATERIAL)
    .has_albedo_map {
      albedo = (albedo * {
        let __cgl_swizzle_4 = sample(*ALBEDO_MAP, uv);
        Vec3::<f32>::new (__cgl_swizzle_4.x, __cgl_swizzle_4.y,
                          __cgl_swizzle_4.z)
      });
    }
  albedo =
      (albedo * Vec3::<f32>::new (input.color.x, input.color.y, input.color.z));
  let mut metallic : f32 = (*MATERIAL).metallic;
  let mut roughness : f32 = (*MATERIAL).roughness;
  if (*MATERIAL)
    .has_metallic_roughness_map {
      let mr_sample : Vec3<f32> = {
        let __cgl_swizzle_5 = sample(*METALLIC_ROUGHNESS_MAP, uv);
      Vec3::<f32>::new (__cgl_swizzle_5.x, __cgl_swizzle_5.y, __cgl_swizzle_5.z)
    };
  metallic *= mr_sample.z;
  roughness *= mr_sample.y;
}
let mut ao : f32 = (*MATERIAL).ao;
if (*MATERIAL)
  .has_ao_map { ao *= sample(*AO_MAP, uv).x; }
let mut emission : Vec3<f32> = (*MATERIAL).emission;
if (*MATERIAL)
  .has_emission_map {
    emission = (emission * {
      let __cgl_swizzle_6 = sample(*EMISSION_MAP, uv);
      Vec3::<f32>::new (__cgl_swizzle_6.x, __cgl_swizzle_6.y, __cgl_swizzle_6.z)
    });
  }
let mut N : Vec3<f32> = normalize(input.world_normal);
if ((*SETTINGS).enable_normal_mapping && (*MATERIAL).has_normal_map) {
  N = getNormalFromMap(*NORMAL_MAP, uv, input.tbn_matrix,
                       (*MATERIAL).normal_scale);
}
let V : Vec3<f32> = normalize(((*CAMERA).position - input.world_position));
let mut Lo : Vec3<f32> = Vec3::<f32>::new (0.0, 0.0, 0.0);
let mut i : i32 = 0;
while ((i < ACTIVE_LIGHT_COUNT) && (i < MAX_LIGHTS)) {
  let light : LightData = LIGHTS[i as usize];
  let mut L : Vec3<f32> = Default::default();
  if (light.type_ == 0) {
    L = normalize((-light.direction));
  } else {
    L = normalize((light.position - input.world_position));
  }
  let attenuation : f32 = calculateAttenuation(light, input.world_position);
  let radiance : Vec3<f32> = {
    let __cgl_vec_arg_27 = Vec3::<f32>::new ((light.color.x * light.intensity),
                                             (light.color.y * light.intensity),
                                             (light.color.z * light.intensity));
  Vec3::<f32>::new ((__cgl_vec_arg_27.x * attenuation),
                    (__cgl_vec_arg_27.y * attenuation),
                    (__cgl_vec_arg_27.z * attenuation))
};
let mut shadow : f32 = 0.0;
if (((*SETTINGS).enable_shadows && light.cast_shadows) &&
    (i < (*SETTINGS).shadow_cascade_count)) {
  shadow =
      calculateShadow(SHADOW_MAPS[i as usize], input.shadow_coords[i as usize],
                      (*SETTINGS).shadow_bias);
}
Lo = (Lo + {
  let __cgl_vec_arg_28 =
      calculateDirectLighting(N, V, L, albedo, metallic, roughness, radiance);
  let __cgl_vec_arg_29 = (1.0 - shadow);
  Vec3::<f32>::new ((__cgl_vec_arg_28.x * __cgl_vec_arg_29),
                    (__cgl_vec_arg_28.y * __cgl_vec_arg_29),
                    (__cgl_vec_arg_28.z * __cgl_vec_arg_29))
});
i += 1;
}
let mut ambient : Vec3<f32> = Vec3::<f32>::new (0.0, 0.0, 0.0);
if (*SETTINGS)
  .enable_ibl {
    ambient = calculateIBL(N, V, albedo, metallic, roughness, *ENVIRONMENT);
  }
else {
  ambient = { let __cgl_vec_arg_30 = ((*ENVIRONMENT).ambient_color * albedo);
  Vec3::<f32>::new ((__cgl_vec_arg_30.x * ao), (__cgl_vec_arg_30.y * ao),
                    (__cgl_vec_arg_30.z * ao))
};
}
let mut color : Vec3<f32> = ((ambient + Lo) + emission);
if (*SETTINGS)
  .enable_tone_mapping { color = acesToneMapping(color); }
if (*SETTINGS)
  .enable_gamma_correction { color = gammaCorrection(color, 2.2); }
return Vec4::<f32>::new (color.x, color.y, color.z, input.color.w);
}

// CrossGL resource metadata: name=environment_map kind=texture set=0 binding=7
// binding_source=automatic CrossGL Rust limitation: resource environment_map is
// emitted as a compile-only placeholder static, not a rust-gpu resource
// binding; pass real spirv_std resources as #[spirv(...)] entry parameters when
// targeting rust-gpu.
static ENVIRONMENT_MAP : std::sync::LazyLock<TextureCube<f32>> =
                             std::sync::LazyLock::new (|| Default::default());
// CrossGL resource metadata: name=irradiance_map kind=image set=0 binding=0
// binding_source=automatic CrossGL Rust limitation: resource irradiance_map is
// emitted as a compile-only placeholder static, not a rust-gpu resource
// binding; pass real spirv_std resources as #[spirv(...)] entry parameters when
// targeting rust-gpu.
static IRRADIANCE_MAP : std::sync::LazyLock<ImageCube<Vec4<f32>>> =
                            std::sync::LazyLock::new (|| Default::default());
static FACE_INDEX : i32 = 0;
static MIP_LEVEL : i32 = 0;
// Compute Shader
#[cfg_attr(feature = "crossgl_gpu", compute_shader)]
pub fn precompute_environment() -> () {
  let coord : Vec2<i32> = Vec2::<i32>::new ((global_invocation_id().x as i32),
                                            (global_invocation_id().y as i32));
  let size : Vec2<i32> = image_size(*IRRADIANCE_MAP);
  if ((coord.x >= size.x) || (coord.y >= size.y)) {
  }
  let mut uv : Vec2<f32> = {
    let __cgl_vec_arg_32 = {let __cgl_vec_arg_31 = Vec2::<f32>::new (
                                (coord.x as f32), (coord.y as f32));
  Vec2::<f32>::new ((__cgl_vec_arg_31.x + 0.5), (__cgl_vec_arg_31.y + 0.5))
};
let __cgl_vec_arg_33 = Vec2::<f32>::new ((size.x as f32), (size.y as f32));
Vec2::<f32>::new ((__cgl_vec_arg_32.x / __cgl_vec_arg_33.x),
                  (__cgl_vec_arg_32.y / __cgl_vec_arg_33.y))
}
;
uv = { let __cgl_vec_arg_34 = Vec2::<f32>::new ((uv.x * 2.0), (uv.y * 2.0));
Vec2::<f32>::new ((__cgl_vec_arg_34.x - 1.0), (__cgl_vec_arg_34.y - 1.0))
}
;
let N : Vec3<f32> = getSamplingVector(uv, FACE_INDEX);
let mut irradiance : Vec3<f32> = Vec3::<f32>::new (0.0, 0.0, 0.0);
let mut sample_count : f32 = 0.0;
let mut phi : f32 = 0.0;
while (phi < (2.0 * PI)) {
  let mut theta : f32 = 0.0;
  while (theta < (0.5 * PI)) {
    let tangent_sample
        : Vec3<f32> = Vec3::<f32>::new ((sin(theta) * cos(phi)),
                                        (sin(theta) * sin(phi)), cos(theta));
    let mut up : Vec3<f32> = (if (abs(N.z) < 0.999) {
      Vec3::<f32>::new (0.0, 0.0, 1.0)
    } else {Vec3::<f32>::new (1.0, 0.0, 0.0)});
    let right : Vec3<f32> = normalize(cross(up, N));
    up = normalize(cross(N, right));
    let sample_vec : Vec3<f32> =
                         ((Vec3::<f32>::new ((tangent_sample.x * right.x),
                                             (tangent_sample.x * right.y),
                                             (tangent_sample.x * right.z)) +
                           Vec3::<f32>::new ((tangent_sample.y * up.x),
                                             (tangent_sample.y * up.y),
                                             (tangent_sample.y * up.z))) +
                          Vec3::<f32>::new ((tangent_sample.z * N.x),
                                            (tangent_sample.z * N.y),
                                            (tangent_sample.z * N.z)));
            irradiance = (irradiance + {
      let __cgl_vec_arg_37 = {
        let __cgl_vec_arg_35 = sample(*ENVIRONMENT_MAP, sample_vec);
      let __cgl_vec_arg_36 = cos(theta);
      Vec3::<f32>::new ((__cgl_vec_arg_35.x * __cgl_vec_arg_36),
                        (__cgl_vec_arg_35.y * __cgl_vec_arg_36),
                        (__cgl_vec_arg_35.z * __cgl_vec_arg_36)) }; let __cgl_vec_arg_38 = sin(theta); Vec3::<f32>::new((__cgl_vec_arg_37.x * __cgl_vec_arg_38), (__cgl_vec_arg_37.y * __cgl_vec_arg_38), (__cgl_vec_arg_37.z * __cgl_vec_arg_38))
  });
  sample_count += 1.0;
  theta += 0.025;
}
phi += 0.025;
}
irradiance = {
  let __cgl_vec_arg_39 = Vec3::<f32>::new (
      (PI * irradiance.x), (PI * irradiance.y), (PI * irradiance.z));
Vec3::<f32>::new ((__cgl_vec_arg_39.x / sample_count),
                  (__cgl_vec_arg_39.y / sample_count),
                  (__cgl_vec_arg_39.z / sample_count))
}
;
image_store(*IRRADIANCE_MAP, Vec3::<i32>::new (coord.x, coord.y, FACE_INDEX),
            Vec4::<f32>::new (irradiance.x, irradiance.y, irradiance.z, 1.0));
}

pub fn getSamplingVector(uv : Vec2<f32>, face : i32) -> Vec3<f32> {
  let mut result : Vec3<f32> = Default::default();
  match face {
    0 => { result = Vec3::<f32>::new (1.0, (-uv.y), (-uv.x)); }
    , 1 => { result = Vec3::<f32>::new ((-1.0), (-uv.y), uv.x); }
    , 2 => { result = Vec3::<f32>::new (uv.x, 1.0, uv.y); }
    , 3 => { result = Vec3::<f32>::new (uv.x, (-1.0), (-uv.y)); }
    , 4 => { result = Vec3::<f32>::new (uv.x, (-uv.y), 1.0); }
    , 5 => { result = Vec3::<f32>::new ((-uv.x), (-uv.y), (-1.0)); }
    , _ =>{},
  }
  return normalize(result);
}
