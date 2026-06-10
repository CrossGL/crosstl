
#include <metal_stdlib>
using namespace metal;

__attribute__((unused)) constant float PI = 3.14159265359;
__attribute__((unused)) constant float EPSILON = 0.0001;
__attribute__((unused)) constant int MAX_LIGHTS = 32;
__attribute__((unused)) constant int MAX_SHADOW_CASCADES = 4;

struct MaterialProperties {
  float3 albedo;
  float metallic;
  float roughness;
  float ao;
  float3 emission;
  float normal_scale;
  float height_scale;
  bool has_albedo_map;
  bool has_normal_map;
  bool has_metallic_roughness_map;
  bool has_ao_map;
  bool has_emission_map;
  bool has_height_map;
};
struct LightData {
  float3 position;
  float3 direction;
  float3 color;
  float intensity;
  float range;
  float inner_cone_angle;
  float outer_cone_angle;
  int type;
  bool cast_shadows;
  float4x4 light_view_proj;
};
struct EnvironmentData {
  texturecube<float> irradiance_map;
  texturecube<float> prefilter_map;
  texture2d<float> brdf_lut;
  float max_reflection_lod;
  float exposure;
  float3 ambient_color;
};
struct CameraData {
  float3 position;
  float3 forward;
  float3 up;
  float3 right;
  float4x4 view_matrix;
  float4x4 projection_matrix;
  float4x4 view_projection_matrix;
  float near_plane;
  float far_plane;
  float fov;
  float2 screen_size;
};
struct RenderSettings {
  bool enable_ibl;
  bool enable_shadows;
  bool enable_normal_mapping;
  bool enable_parallax_mapping;
  bool enable_tone_mapping;
  bool enable_gamma_correction;
  int shadow_cascade_count;
  float shadow_bias;
  int max_lights;
  float lod_bias;
};
struct VertexInput {
  float3 position [[attribute(0)]];
  float3 normal [[attribute(1)]];
  float3 tangent [[attribute(2)]];
  float2 uv [[attribute(3)]];
  float4 color [[attribute(4)]];
};
struct VertexOutput {
  float4 clip_position [[position]];
  float3 world_position;
  float3 world_normal;
  float3 world_tangent;
  float3 world_bitangent;
  float2 uv;
  float4 color;
  float3 tbn_matrix_0;
  float3 tbn_matrix_1;
  float3 tbn_matrix_2;
  float4 shadow_coords_0;
  float4 shadow_coords_1;
  float4 shadow_coords_2;
  float4 shadow_coords_3;
};
struct FragmentInput {
  float3 world_position;
  float3 world_normal;
  float3 world_tangent;
  float3 world_bitangent;
  float2 uv;
  float4 color;
  float3 tbn_matrix_0;
  float3 tbn_matrix_1;
  float3 tbn_matrix_2;
  float4 shadow_coords_0;
  float4 shadow_coords_1;
  float4 shadow_coords_2;
  float4 shadow_coords_3;
};
static inline float4 __attribute__((unused))
__crossgl_stage_io_get_FragmentInput_shadow_coords(FragmentInput value,
                                                   int index) {
  switch (index) {
  case 0:
    return value.shadow_coords_0;
  case 1:
    return value.shadow_coords_1;
  case 2:
    return value.shadow_coords_2;
  case 3:
    return value.shadow_coords_3;
  default:
    return float4(0);
  }
}

static inline float4 __attribute__((unused))
__crossgl_stage_io_get_VertexOutput_shadow_coords(VertexOutput value,
                                                  int index) {
  switch (index) {
  case 0:
    return value.shadow_coords_0;
  case 1:
    return value.shadow_coords_1;
  case 2:
    return value.shadow_coords_2;
  case 3:
    return value.shadow_coords_3;
  default:
    return float4(0);
  }
}

float3 getNormalFromMap(texture2d<float> normal_map, float2 uv, float3x3 tbn,
                        float scale) {
  float3 tangent_normal =
      normal_map.sample(sampler(mag_filter::linear, min_filter::linear), uv)
              .xyz *
          float3(2.0) -
      float3(1.0);
  tangent_normal.xy *= scale;
  return normalize(tbn * tangent_normal);
}

float2 parallaxMapping(texture2d<float> height_map, float2 uv, float3 view_dir,
                       float height_scale) {
  float height =
      height_map.sample(sampler(mag_filter::linear, min_filter::linear), uv).r;
  float2 p = view_dir.xy / float2(view_dir.z) * float2(height * height_scale);
  return uv - p;
}

float distributionGGX(float3 N, float3 H, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float NdotH = max(dot(N, H), 0.0);
  float NdotH2 = NdotH * NdotH;
  float num = a2;
  float denom = NdotH2 * a2 - 1.0 + 1.0;
  denom = PI * denom * denom;
  return num / max(denom, EPSILON);
}

float geometrySchlickGGX(float NdotV, float roughness) {
  float r = roughness + 1.0;
  float k = r * r / 8.0;
  float num = NdotV;
  float denom = NdotV * 1.0 - k + k;
  return num / max(denom, EPSILON);
}

float geometrySmith(float3 N, float3 V, float3 L, float roughness) {
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float ggx2 = geometrySchlickGGX(NdotV, roughness);
  float ggx1 = geometrySchlickGGX(NdotL, roughness);
  return ggx1 * ggx2;
}

float3 fresnelSchlick(float cosTheta, float3 F0) {
  return F0 + float3(1.0) - F0 * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness) {
  return F0 + max(float3(1.0 - roughness), F0) -
         F0 * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float calculateShadow(texture2d<float> shadow_map, float4 frag_pos_light_space,
                      float bias) {
  float3 proj_coords =
      frag_pos_light_space.xyz / float3(frag_pos_light_space.w);
  proj_coords = proj_coords * float3(0.5) + float3(0.5);
  if (proj_coords.z > 1.0) {
    return 0.0;
  }
  float shadow = 0.0;
  float2 texel_size =
      float2(1.0) / float2(int2(shadow_map.get_width(uint(0)),
                                shadow_map.get_height(uint(0))));
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      float pcf_depth =
          shadow_map
              .sample(sampler(mag_filter::linear, min_filter::linear),
                      proj_coords.xy + float2(x, y) * texel_size)
              .r;
      shadow += proj_coords.z - bias > pcf_depth ? 1.0 : 0.0;
    }
  }
  return shadow / 9.0;
}

float3 calculateIBL(float3 N, float3 V, float3 albedo, float metallic,
                    float roughness, EnvironmentData env) {
  float3 F0 = mix(float3(0.04), albedo, metallic);
  float3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
  float3 kS = F;
  float3 kD = float3(1.0) - kS;
  kD *= 1.0 - metallic;
  float3 irradiance =
      env.irradiance_map
          .sample(sampler(mag_filter::linear, min_filter::linear), N)
          .rgb;
  float3 diffuse = irradiance * albedo;
  float3 R = reflect(-V, N);
  float3 prefiltered_color =
      env.prefilter_map
          .sample(sampler(mag_filter::linear, min_filter::linear), R,
                  level(roughness * env.max_reflection_lod))
          .rgb;
  float2 brdf = env.brdf_lut
                    .sample(sampler(mag_filter::linear, min_filter::linear),
                            float2(max(dot(N, V), 0.0), roughness))
                    .rg;
  float3 specular = prefiltered_color * F * float3(brdf.x) + float3(brdf.y);
  return kD * diffuse + specular * float3(env.exposure);
}

float3 calculateDirectLighting(float3 N, float3 V, float3 L, float3 albedo,
                               float metallic, float roughness,
                               float3 light_color) {
  float3 H = normalize(V + L);
  float3 F0 = mix(float3(0.04), albedo, metallic);
  float NDF = distributionGGX(N, H, roughness);
  float G = geometrySmith(N, V, L, roughness);
  float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
  float3 kS = F;
  float3 kD = float3(1.0) - kS;
  kD *= 1.0 - metallic;
  float3 numerator = float3(NDF * G) * F;
  float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + EPSILON;
  float3 specular = numerator / float3(denominator);
  float NdotL = max(dot(N, L), 0.0);
  return kD * albedo / PI + specular * light_color * float3(NdotL);
}

float calculateAttenuation(LightData light, float3 frag_pos) {
  if (light.type == 0) {
    return 1.0;
  }
  float distance = length(light.position - frag_pos);
  if (light.type == 1) {
    float attenuation = 1.0 / distance * distance;
    return attenuation * smoothstep(light.range, 0.0, distance);
  }
  if (light.type == 2) {
    float3 light_dir = normalize(light.position - frag_pos);
    float theta = dot(light_dir, normalize(-light.direction));
    float epsilon = light.inner_cone_angle - light.outer_cone_angle;
    float intensity = clamp(theta - light.outer_cone_angle / epsilon, 0.0, 1.0);
    float attenuation = 1.0 / distance * distance;
    return attenuation * intensity * smoothstep(light.range, 0.0, distance);
  }
  return 0.0;
}

float3 reinhardToneMapping(float3 color) { return color / color + float3(1.0); }

float3 acesToneMapping(float3 color) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp(color * float3(a) * color +
                   float3(b) / color * float3(c) * color + float3(d) +
                   float3(e),
               0.0, 1.0);
}

float3 gammaCorrection(float3 color, float gamma) {
  return pow(color, float3(1.0 / gamma));
}

// Vertex Shader
vertex VertexOutput vertex_main(
    VertexInput input [[stage_in]],
    constant float4x4 &model_matrix [[buffer(0)]],
    constant float4x4 &view_matrix [[buffer(1)]],
    constant float4x4 &projection_matrix [[buffer(2)]],
    constant float3x3 &normal_matrix [[buffer(3)]],
    constant CameraData &camera [[buffer(4)]],
    constant RenderSettings &settings [[buffer(5)]],
    constant float4x4 *shadow_matrices [[buffer(6)]],
    constant MaterialProperties &material [[buffer(7)]],
    constant EnvironmentData &environment [[buffer(8)]],
    constant LightData *lights [[buffer(9)]],
    constant int &active_light_count [[buffer(10)]],
    constant int &face_index [[buffer(11)]],
    constant int &mip_level [[buffer(12)]],
    texture2d<float> albedo_map [[texture(0)]],
    texture2d<float> normal_map [[texture(1)]],
    texture2d<float> metallic_roughness_map [[texture(2)]],
    texture2d<float> ao_map [[texture(3)]],
    texture2d<float> emission_map [[texture(4)]],
    texture2d<float> height_map [[texture(5)]],
    array<texture2d<float>, MAX_SHADOW_CASCADES> shadow_maps [[texture(6)]],
    texturecube<float> environment_map [[texture(10)]]) {
  VertexOutput output;
  float4 world_pos = model_matrix * float4(input.position, 1.0);
  output.world_position = world_pos.xyz;
  output.clip_position = camera.view_projection_matrix * world_pos;
  output.world_normal = normalize(normal_matrix * input.normal);
  output.world_tangent = normalize(normal_matrix * input.tangent);
  output.world_bitangent = cross(output.world_normal, output.world_tangent);
  float3x3 __crossgl_stage_io_matrix_0 = float3x3(
      output.world_tangent, output.world_bitangent, output.world_normal);
  output.tbn_matrix_0 = __crossgl_stage_io_matrix_0[0];
  output.tbn_matrix_1 = __crossgl_stage_io_matrix_0[1];
  output.tbn_matrix_2 = __crossgl_stage_io_matrix_0[2];
  output.uv = input.uv;
  output.color = input.color;
  if (settings.enable_shadows) {
    for (int i = 0;
         i < settings.shadow_cascade_count && i < MAX_SHADOW_CASCADES; ++i) {
      float4 __crossgl_stage_io_array_1 = shadow_matrices[i] * world_pos;
      switch (i) {
      case 0:
        output.shadow_coords_0 = __crossgl_stage_io_array_1;
        break;
      case 1:
        output.shadow_coords_1 = __crossgl_stage_io_array_1;
        break;
      case 2:
        output.shadow_coords_2 = __crossgl_stage_io_array_1;
        break;
      case 3:
        output.shadow_coords_3 = __crossgl_stage_io_array_1;
        break;
      default:
        break;
      }
    }
  }
  return output;
}

// Fragment Shader
fragment float4 fragment_main(
    FragmentInput input [[stage_in]],
    constant float4x4 &model_matrix [[buffer(0)]],
    constant float4x4 &view_matrix [[buffer(1)]],
    constant float4x4 &projection_matrix [[buffer(2)]],
    constant float3x3 &normal_matrix [[buffer(3)]],
    constant CameraData &camera [[buffer(4)]],
    constant RenderSettings &settings [[buffer(5)]],
    constant float4x4 *shadow_matrices [[buffer(6)]],
    constant MaterialProperties &material [[buffer(7)]],
    constant EnvironmentData &environment [[buffer(8)]],
    constant LightData *lights [[buffer(9)]],
    constant int &active_light_count [[buffer(10)]],
    constant int &face_index [[buffer(11)]],
    constant int &mip_level [[buffer(12)]],
    texture2d<float> albedo_map [[texture(0)]],
    texture2d<float> normal_map [[texture(1)]],
    texture2d<float> metallic_roughness_map [[texture(2)]],
    texture2d<float> ao_map [[texture(3)]],
    texture2d<float> emission_map [[texture(4)]],
    texture2d<float> height_map [[texture(5)]],
    array<texture2d<float>, MAX_SHADOW_CASCADES> shadow_maps [[texture(6)]],
    texturecube<float> environment_map [[texture(10)]],
    texture2d_array<float, access::read_write> irradiance_map [[texture(11)]]) {
  float2 uv = input.uv;
  if (settings.enable_parallax_mapping && material.has_height_map) {
    float3 view_dir = normalize(camera.position - input.world_position);
    float3 tangent_view_dir =
        transpose(float3x3(input.tbn_matrix_0, input.tbn_matrix_1,
                           input.tbn_matrix_2)) *
        view_dir;
    uv = parallaxMapping(height_map, uv, tangent_view_dir,
                         material.height_scale);
  }
  float3 albedo = material.albedo;
  if (material.has_albedo_map) {
    albedo *=
        albedo_map.sample(sampler(mag_filter::linear, min_filter::linear), uv)
            .rgb;
  }
  albedo *= input.color.rgb;
  float metallic = material.metallic;
  float roughness = material.roughness;
  if (material.has_metallic_roughness_map) {
    float3 mr_sample =
        metallic_roughness_map
            .sample(sampler(mag_filter::linear, min_filter::linear), uv)
            .rgb;
    metallic *= mr_sample.b;
    roughness *= mr_sample.g;
  }
  float ao = material.ao;
  if (material.has_ao_map) {
    ao *= ao_map.sample(sampler(mag_filter::linear, min_filter::linear), uv).r;
  }
  float3 emission = material.emission;
  if (material.has_emission_map) {
    emission *=
        emission_map.sample(sampler(mag_filter::linear, min_filter::linear), uv)
            .rgb;
  }
  float3 N = normalize(input.world_normal);
  if (settings.enable_normal_mapping && material.has_normal_map) {
    N = getNormalFromMap(
        normal_map, uv,
        float3x3(input.tbn_matrix_0, input.tbn_matrix_1, input.tbn_matrix_2),
        material.normal_scale);
  }
  float3 V = normalize(camera.position - input.world_position);
  float3 Lo = float3(0.0);
  for (int i = 0; i < active_light_count && i < MAX_LIGHTS; ++i) {
    LightData light = lights[i];
    float3 L;
    if (light.type == 0) {
      L = normalize(-light.direction);
    } else {
      L = normalize(light.position - input.world_position);
    }
    float attenuation = calculateAttenuation(light, input.world_position);
    float3 radiance =
        light.color * float3(light.intensity) * float3(attenuation);
    float shadow = 0.0;
    if (settings.enable_shadows && light.cast_shadows &&
        i < settings.shadow_cascade_count) {
      shadow = calculateShadow(
          shadow_maps[i],
          __crossgl_stage_io_get_FragmentInput_shadow_coords(input, i),
          settings.shadow_bias);
    }
    Lo += calculateDirectLighting(N, V, L, albedo, metallic, roughness,
                                  radiance) *
          float3(1.0 - shadow);
  }
  float3 ambient = float3(0.0);
  if (settings.enable_ibl) {
    ambient = calculateIBL(N, V, albedo, metallic, roughness, environment);
  } else {
    ambient = environment.ambient_color * albedo * float3(ao);
  }
  float3 color = ambient + Lo + emission;
  if (settings.enable_tone_mapping) {
    color = acesToneMapping(color);
  }
  if (settings.enable_gamma_correction) {
    color = gammaCorrection(color, 2.2);
  }
  return float4(color, input.color.a);
}

float3 getSamplingVector(float2 uv, int face) {
  float3 result;
  switch (face) {
  case 0: {
    result = float3(1.0, -uv.y, -uv.x);
    break;
  }
  case 1: {
    result = float3(-1.0, -uv.y, uv.x);
    break;
  }
  case 2: {
    result = float3(uv.x, 1.0, uv.y);
    break;
  }
  case 3: {
    result = float3(uv.x, -1.0, -uv.y);
    break;
  }
  case 4: {
    result = float3(uv.x, -uv.y, 1.0);
    break;
  }
  case 5: {
    result = float3(-uv.x, -uv.y, -1.0);
    break;
  }
  }
  return normalize(result);
}

// Compute Shader
kernel void kernel_precompute_environment(
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    constant float4x4 &model_matrix [[buffer(0)]],
    constant float4x4 &view_matrix [[buffer(1)]],
    constant float4x4 &projection_matrix [[buffer(2)]],
    constant float3x3 &normal_matrix [[buffer(3)]],
    constant CameraData &camera [[buffer(4)]],
    constant RenderSettings &settings [[buffer(5)]],
    constant float4x4 *shadow_matrices [[buffer(6)]],
    constant MaterialProperties &material [[buffer(7)]],
    constant EnvironmentData &environment [[buffer(8)]],
    constant LightData *lights [[buffer(9)]],
    constant int &active_light_count [[buffer(10)]],
    constant int &face_index [[buffer(11)]],
    constant int &mip_level [[buffer(12)]],
    texture2d<float> albedo_map [[texture(0)]],
    texture2d<float> normal_map [[texture(1)]],
    texture2d<float> metallic_roughness_map [[texture(2)]],
    texture2d<float> ao_map [[texture(3)]],
    texture2d<float> emission_map [[texture(4)]],
    texture2d<float> height_map [[texture(5)]],
    array<texture2d<float>, MAX_SHADOW_CASCADES> shadow_maps [[texture(6)]],
    texturecube<float> environment_map [[texture(10)]],
    texture2d_array<float, access::read_write> irradiance_map [[texture(11)]]) {
  int2 coord = int2(thread_position_in_grid.xy);
  int2 size = int2(irradiance_map.get_width(), irradiance_map.get_height());
  if (coord.x >= size.x || coord.y >= size.y) {
    return;
  }
  float2 uv = float2(coord) + float2(0.5) / float2(size);
  uv = uv * float2(2.0) - float2(1.0);
  float3 N = getSamplingVector(uv, face_index);
  float3 irradiance = float3(0.0);
  float sample_count = 0.0;
  for (float phi = 0.0; phi < 2.0 * PI; phi += 0.025) {
    for (float theta = 0.0; theta < 0.5 * PI; theta += 0.025) {
      float3 tangent_sample =
          float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
      float3 up =
          abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
      float3 right = normalize(cross(up, N));
      up = normalize(cross(N, right));
      float3 sample_vec = float3(tangent_sample.x) * right +
                          float3(tangent_sample.y) * up +
                          float3(tangent_sample.z) * N;
      irradiance += environment_map
                        .sample(sampler(mag_filter::linear, min_filter::linear),
                                sample_vec)
                        .rgb *
                    cos(theta) * sin(theta);
      ++sample_count;
    }
  }
  irradiance = PI * irradiance / float3(sample_count);
  irradiance_map.write(float4(irradiance, 1.0),
                       uint2((int3(coord, face_index)).xy),
                       uint((int3(coord, face_index)).z));
}
