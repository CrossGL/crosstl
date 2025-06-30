// Generated Rust GPU Shader Code
use gpu::*;
use math::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MaterialProperties {
  pub albedo : Vec3<f32>,
               pub metallic : f32,
                              pub roughness : f32,
                                              pub ao : f32,
                                                       pub emission
      : Vec3<f32>,
        pub normal_scale : f32,
                           pub height_scale : f32,
                                              pub has_albedo_map
      : bool,
        pub has_normal_map : bool,
                             pub has_metallic_roughness_map : bool,
                                                              pub has_ao_map
      : bool,
        pub has_emission_map : bool,
                               pub has_height_map : bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LightData {
  pub position : Vec3<f32>,
                 pub direction : Vec3<f32>,
                                 pub color : Vec3<f32>,
                                             pub intensity : f32,
                                                             pub range
      : f32,
        pub inner_cone_angle : f32,
                               pub outer_cone_angle : f32,
                                                      pub type
      : i32,
        pub cast_shadows : bool,
                           pub light_view_proj : mat4x4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EnvironmentData {
  pub irradiance_map : TextureCube<f32>,
                       pub prefilter_map : TextureCube<f32>,
                                           pub brdf_lut : Texture2D<f32>,
                                                          pub max_reflection_lod
      : f32,
        pub exposure : f32,
                       pub ambient_color : Vec3<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CameraData {
  pub position : Vec3<f32>,
                 pub forward : Vec3<f32>,
                               pub up : Vec3<f32>,
                                        pub right : Vec3<f32>,
                                                    pub view_matrix
      : mat4x4,
        pub projection_matrix : mat4x4,
                                pub view_projection_matrix : mat4x4,
                                                             pub near_plane
      : f32,
        pub far_plane : f32,
                        pub fov : f32,
                                  pub screen_size : Vec2<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RenderSettings {
  pub enable_ibl : bool,
                   pub enable_shadows : bool,
                                        pub enable_normal_mapping
      : bool,
        pub enable_parallax_mapping : bool,
                                      pub enable_tone_mapping
      : bool,
        pub enable_gamma_correction : bool,
                                      pub shadow_cascade_count : i32,
                                                                 pub shadow_bias
      : f32,
        pub max_lights : i32,
                         pub lod_bias : f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexOutput {
  pub clip_position : Vec4<f32>,
                      pub world_position : Vec3<f32>,
                                           pub world_normal : Vec3<f32>,
                                                              pub world_tangent
      : Vec3<f32>,
        pub world_bitangent : Vec3<f32>,
                              pub uv : Vec2<f32>,
                                       pub color : Vec4<f32>,
                                                   pub tbn_matrix
      : mat3x3,
        pub shadow_coords : vec4IdentifierNode(name = MAX_SHADOW_CASCADES),
}

static model_matrix : mat4x4 = Default::default();
static view_matrix : mat4x4 = Default::default();
static projection_matrix : mat4x4 = Default::default();
static normal_matrix : mat3x3 = Default::default();
static camera : CameraData = Default::default();
static settings : RenderSettings = Default::default();
static shadow_matrices
    : mat4x4IdentifierNode(name = MAX_SHADOW_CASCADES) = Default::default();
static albedo_map : Texture2D<f32> = Default::default();
static normal_map : Texture2D<f32> = Default::default();
static metallic_roughness_map : Texture2D<f32> = Default::default();
static ao_map : Texture2D<f32> = Default::default();
static emission_map : Texture2D<f32> = Default::default();
static height_map : Texture2D<f32> = Default::default();
static shadow_maps
    : sampler2DIdentifierNode(name = MAX_SHADOW_CASCADES) = Default::default();
static material : MaterialProperties = Default::default();
static environment : EnvironmentData = Default::default();
static camera : CameraData = Default::default();
static settings : RenderSettings = Default::default();
static lights : LightDataIdentifierNode(name = MAX_LIGHTS) = Default::default();
static active_light_count : i32 = Default::default();
// Constant Buffers
pub fn getNormalFromMap(normal_map : Texture2D<f32>, uv : Vec2<f32>,
                        tbn : mat3x3, scale : f32) -> Vec3<f32> {
  let mut tangent_normal : Vec3<f32> =
                               ((sample(normal_map, uv).xyz * 2.0) - 1.0);
  tangent_normal.xy *= scale;
  return normalize((tbn * tangent_normal));
}

pub fn parallaxMapping(height_map : Texture2D<f32>, uv : Vec2<f32>,
                       view_dir : Vec3<f32>, height_scale : f32) -> Vec2<f32> {
  let mut height : f32 = sample(height_map, uv).r;
  let mut p : Vec2<f32> =
                  ((view_dir.xy / view_dir.z) * (height * height_scale));
  return (uv - p);
}

pub fn distributionGGX(N : Vec3<f32>, H : Vec3<f32>, roughness : f32) -> f32 {
  let mut a : f32 = (roughness * roughness);
  let mut a2 : f32 = (a * a);
  let mut NdotH : f32 = max(dot(N, H), 0.0);
  let mut NdotH2 : f32 = (NdotH * NdotH);
  let mut num : f32 = a2;
  let mut denom : f32 = ((NdotH2 * (a2 - 1.0)) + 1.0);
  denom = ((PI * denom) * denom);
  return (num / max(denom, EPSILON));
}

pub fn geometrySchlickGGX(NdotV : f32, roughness : f32) -> f32 {
  let mut r : f32 = (roughness + 1.0);
  let mut k : f32 = ((r * r) / 8.0);
  let mut num : f32 = NdotV;
  let mut denom : f32 = ((NdotV * (1.0 - k)) + k);
  return (num / max(denom, EPSILON));
}

pub fn geometrySmith(N : Vec3<f32>, V : Vec3<f32>, L : Vec3<f32>,
                     roughness : f32) -> f32 {
  let mut NdotV : f32 = max(dot(N, V), 0.0);
  let mut NdotL : f32 = max(dot(N, L), 0.0);
  let mut ggx2 : f32 = geometrySchlickGGX(NdotV, roughness);
  let mut ggx1 : f32 = geometrySchlickGGX(NdotL, roughness);
  return (ggx1 * ggx2);
}

pub fn fresnelSchlick(cosTheta : f32, F0 : Vec3<f32>) -> Vec3<f32> {
  return (F0 + ((1.0 - F0) * pow(max((1.0 - cosTheta), 0.0), 5.0)));
}

pub fn fresnelSchlickRoughness(cosTheta : f32, F0 : Vec3<f32>, roughness : f32)
    -> Vec3<f32> {
  return (F0 + ((max(Vec3<f32>::new ((1.0 - roughness)), F0) - F0) *
                pow(max((1.0 - cosTheta), 0.0), 5.0)));
}

pub fn calculateShadow(shadow_map : Texture2D<f32>,
                       frag_pos_light_space : Vec4<f32>, bias : f32) -> f32 {
  let mut proj_coords : Vec3<f32> =
                            (frag_pos_light_space.xyz / frag_pos_light_space.w);
  proj_coords = ((proj_coords * 0.5) + 0.5);
  if (proj_coords.z > 1.0) {
  }
  let mut shadow : f32 = 0.0;
  let mut texel_size : Vec2<f32> = (1.0 / textureSize(shadow_map, 0));
  let mut x : i32 = (-1);
  ;
  while (x <= 1) {
    let mut y : i32 = (-1);
    ;
    while (y <= 1) {
      let mut pcf_depth
          : f32 =
                sample(shadow_map,
                       (proj_coords.xy + (Vec2<f32>::new (x, y) * texel_size)))
                    .r;
      shadow += (if ((proj_coords.z - bias) > pcf_depth){1.0} else {0.0});
      (++y);
    }
    (++x);
  }
  return (shadow / 9.0);
}

pub fn calculateIBL(N : Vec3<f32>, V : Vec3<f32>, albedo : Vec3<f32>,
                    metallic : f32, roughness : f32, env : EnvironmentData)
    -> Vec3<f32> {
  let mut F0 : Vec3<f32> = lerp(Vec3<f32>::new (0.04), albedo, metallic);
  let mut F : Vec3<f32> =
                  fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
  let mut kS : Vec3<f32> = F;
  let mut kD : Vec3<f32> = (1.0 - kS);
  kD *= (1.0 - metallic);
  let mut irradiance : Vec3<f32> = sample(env.irradiance_map, N).rgb;
  let mut diffuse : Vec3<f32> = (irradiance * albedo);
  let mut R : Vec3<f32> = reflect((-V), N);
  let mut prefiltered_color
      : Vec3<f32> = textureLod(env.prefilter_map, R,
                               (roughness * env.max_reflection_lod))
                        .rgb;
  let mut brdf : Vec2<f32> =
                     sample(env.brdf_lut,
                            Vec2<f32>::new (max(dot(N, V), 0.0), roughness))
                         .rg;
  let mut specular : Vec3<f32> = (prefiltered_color * ((F * brdf.x) + brdf.y));
  return (((kD * diffuse) + specular) * env.exposure);
}

pub fn calculateDirectLighting(N : Vec3<f32>, V : Vec3<f32>, L : Vec3<f32>,
                               albedo : Vec3<f32>, metallic : f32,
                               roughness : f32, light_color : Vec3<f32>)
    -> Vec3<f32> {
  let mut H : Vec3<f32> = normalize((V + L));
  let mut F0 : Vec3<f32> = lerp(Vec3<f32>::new (0.04), albedo, metallic);
  let mut NDF : f32 = distributionGGX(N, H, roughness);
  let mut G : f32 = geometrySmith(N, V, L, roughness);
  let mut F : Vec3<f32> = fresnelSchlick(max(dot(H, V), 0.0), F0);
  let mut kS : Vec3<f32> = F;
  let mut kD : Vec3<f32> = (Vec3<f32>::new (1.0) - kS);
  kD *= (1.0 - metallic);
  let mut numerator : Vec3<f32> = ((NDF * G) * F);
  let mut denominator
      : f32 = (((4.0 * max(dot(N, V), 0.0)) * max(dot(N, L), 0.0)) + EPSILON);
  let mut specular : Vec3<f32> = (numerator / denominator);
  let mut NdotL : f32 = max(dot(N, L), 0.0);
  return (((((kD * albedo) / PI) + specular) * light_color) * NdotL);
}

pub fn calculateAttenuation(light : LightData, frag_pos : Vec3<f32>) -> f32 {
  if (light.type == 0) {
    return 1.0;
  }
  let mut distance : f32 = length((light.position - frag_pos));
  if (light.type == 1) {
    let mut attenuation : f32 = (1.0 / (distance * distance));
    return (attenuation * smoothstep(light.range, 0.0, distance));
  }
  if (light.type == 2) {
    let mut light_dir : Vec3<f32> = normalize((light.position - frag_pos));
    let mut theta : f32 = dot(light_dir, normalize((-light.direction)));
    let mut epsilon : f32 = (light.inner_cone_angle - light.outer_cone_angle);
    let mut intensity
        : f32 = clamp(((theta - light.outer_cone_angle) / epsilon), 0.0, 1.0);
    let mut attenuation : f32 = (1.0 / (distance * distance));
    return ((attenuation * intensity) * smoothstep(light.range, 0.0, distance));
  }
  return 0.0;
}

pub fn reinhardToneMapping(color : Vec3<f32>) -> Vec3<f32> {
  return (color / (color + Vec3<f32>::new (1.0)));
}

pub fn acesToneMapping(color : Vec3<f32>) -> Vec3<f32> {
  let mut a : f32 = 2.51;
  let mut b : f32 = 0.03;
  let mut c : f32 = 2.43;
  let mut d : f32 = 0.59;
  let mut e : f32 = 0.14;
  return clamp(
      ((color * ((a * color) + b)) / ((color * ((c * color) + d)) + e)), 0.0,
      1.0);
}

pub fn gammaCorrection(color : Vec3<f32>, gamma : f32) -> Vec3<f32> {
  return pow(color, Vec3<f32>::new ((1.0 / gamma)));
}

pub fn main(input : VertexInput) -> VertexOutput {
  let mut output : VertexOutput;
  let mut world_pos : Vec4<f32> =
                          (model_matrix * Vec4<f32>::new (input.position, 1.0));
  output.world_position = world_pos.xyz;
  output.clip_position = (camera.view_projection_matrix * world_pos);
  output.world_normal = normalize((normal_matrix * input.normal));
  output.world_tangent = normalize((normal_matrix * input.tangent));
  output.world_bitangent = cross(output.world_normal, output.world_tangent);
  output.tbn_matrix = Mat3<f32>::new (
      output.world_tangent, output.world_bitangent, output.world_normal);
  output.uv = input.uv;
  output.color = input.color;
  if settings
    .enable_shadows {
      let mut i : i32 = 0;
      ;
      while ((i < settings.shadow_cascade_count) && (i < MAX_SHADOW_CASCADES)) {
        output.shadow_coords[i] = (shadow_matrices[i] * world_pos);
        (++i);
      }
    }
  return output;
}

pub fn main(input : FragmentInput) -> Vec4<f32> {
  let mut uv : Vec2<f32> = input.uv;
  if (settings.enable_parallax_mapping && material.has_height_map) {
    let mut view_dir : Vec3<f32> =
                           normalize((camera.position - input.world_position));
    let mut tangent_view_dir : Vec3<f32> =
                                   (transpose(input.tbn_matrix) * view_dir);
    uv = parallaxMapping(height_map, uv, tangent_view_dir,
                         material.height_scale);
  }
  let mut albedo : Vec3<f32> = material.albedo;
  if material
    .has_albedo_map { albedo *= sample(albedo_map, uv).rgb; }
  albedo *= input.color.rgb;
  let mut metallic : f32 = material.metallic;
  let mut roughness : f32 = material.roughness;
  if material
    .has_metallic_roughness_map {
      let mut mr_sample : Vec3<f32> = sample(metallic_roughness_map, uv).rgb;
      metallic *= mr_sample.b;
      roughness *= mr_sample.g;
    }
  let mut ao : f32 = material.ao;
  if material
    .has_ao_map { ao *= sample(ao_map, uv).r; }
  let mut emission : Vec3<f32> = material.emission;
  if material
    .has_emission_map { emission *= sample(emission_map, uv).rgb; }
  let mut N : Vec3<f32> = normalize(input.world_normal);
  if (settings.enable_normal_mapping && material.has_normal_map) {
    N = getNormalFromMap(normal_map, uv, input.tbn_matrix,
                         material.normal_scale);
  }
  let mut V : Vec3<f32> = normalize((camera.position - input.world_position));
  let mut Lo : Vec3<f32> = Vec3<f32>::new (0.0);
  let mut i : i32 = 0;
  ;
  while ((i < active_light_count) && (i < MAX_LIGHTS)) {
    let mut light : LightData = lights[i];
    let mut L : Vec3<f32>;
    if (light.type == 0) {
      L = normalize((-light.direction));
    } else {
      L = normalize((light.position - input.world_position));
    }
    let mut attenuation : f32 =
                              calculateAttenuation(light, input.world_position);
    let mut radiance : Vec3<f32> =
                           ((light.color * light.intensity) * attenuation);
    let mut shadow : f32 = 0.0;
    if ((settings.enable_shadows && light.cast_shadows) &&
        (i < settings.shadow_cascade_count)) {
      shadow = calculateShadow(shadow_maps[i], input.shadow_coords[i],
                               settings.shadow_bias);
    }
    Lo += (calculateDirectLighting(N, V, L, albedo, metallic, roughness,
                                   radiance) *
           (1.0 - shadow));
    (++i);
  }
  let mut ambient : Vec3<f32> = Vec3<f32>::new (0.0);
  if settings
    .enable_ibl {
      ambient = calculateIBL(N, V, albedo, metallic, roughness, environment);
    }
  else {
    ambient = ((environment.ambient_color * albedo) * ao);
  }
  let mut color : Vec3<f32> = ((ambient + Lo) + emission);
  if settings
    .enable_tone_mapping { color = acesToneMapping(color); }
  if settings
    .enable_gamma_correction { color = gammaCorrection(color, 2.2); }
  return Vec4<f32>::new (color, input.color.a);
}

// Vertex Shader
#[vertex_shader]
pub fn main() -> () {}

// Fragment Shader
#[fragment_shader]
pub fn main() -> () {}

// Compute Shader
#[compute_shader]
pub fn precompute_environment() -> () {
  let mut coord : Vec2<i32> = Vec2<i32>::new (gl_GlobalInvocationID.xy);
  let mut size : Vec2<i32> = imageSize(irradiance_map);
  if ((coord.x >= size.x) || (coord.y >= size.y)) {
  }
  let mut uv : Vec2<f32> =
                   ((Vec2<f32>::new (coord) + 0.5) / Vec2<f32>::new (size));
  uv = ((uv * 2.0) - 1.0);
  let mut N : Vec3<f32> = getSamplingVector(uv, face_index);
  let mut irradiance : Vec3<f32> = Vec3<f32>::new (0.0);
  let mut sample_count : f32 = 0.0;
  let mut phi : f32 = 0.0;
  ;
  while (phi < (2.0 * PI)) {
    let mut theta : f32 = 0.0;
    ;
    while (theta < (0.5 * PI)) {
      let mut tangent_sample
          : Vec3<f32> = Vec3<f32>::new ((sin(theta) * cos(phi)),
                                        (sin(theta) * sin(phi)), cos(theta));
      let mut up : Vec3<f32> = (if (abs(N.z) < 0.999) {
        Vec3<f32>::new (0.0, 0.0, 1.0)
      } else {Vec3<f32>::new (1.0, 0.0, 0.0)});
      let mut right : Vec3<f32> = normalize(cross(up, N));
      up = normalize(cross(N, right));
      let mut sample_vec
          : Vec3<f32> =
                (((tangent_sample.x * right) + (tangent_sample.y * up)) +
                 (tangent_sample.z * N));
      irradiance +=
          ((sample(environment_map, sample_vec).rgb * cos(theta)) * sin(theta));
      (++sample_count);
      theta += 0.025;
    }
    phi += 0.025;
  }
  irradiance = ((PI * irradiance) / sample_count);
  imageStore(irradiance_map, Vec3<i32>::new (coord, face_index),
             Vec4<f32>::new (irradiance, 1.0));
}

pub fn getSamplingVector(uv : Vec2<f32>, face : i32) -> Vec3<f32> {
  let mut result : Vec3<f32>;
  SwitchNode(cases = 6);
  return normalize(result);
}
