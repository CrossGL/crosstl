
#include <metal_stdlib>
using namespace metal;

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
struct VertexOutput {
  float4 clip_position;
  float3 world_position;
  float3 world_normal;
  float3 world_tangent;
  float3 world_bitangent;
  float2 uv;
  float4 color;
  float3x3 tbn_matrix;
  float4 shadow_coords[MAX_SHADOW_CASCADES];
};
float4x4 model_matrix;
float4x4 view_matrix;
float4x4 projection_matrix;
float3x3 normal_matrix;
CameraData camera;
RenderSettings settings;
float4x4 shadow_matrices[MAX_SHADOW_CASCADES];
MaterialProperties material;
EnvironmentData environment;
CameraData camera;
RenderSettings settings;
LightData lights[MAX_LIGHTS];
int active_light_count;
float3 getNormalFromMap(
    texture2d<float> normal_map [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 2) uv [[stage_in]],
    MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
               rows = 3, cols = 3) tbn [[stage_in]],
    float scale [[stage_in]]) {
  float3 tangent_normal =
      normal_map.sample(sampler(mag_filter::linear, min_filter::linear), uv)
              .xyz *
          2.0 -
      1.0;
  tangent_normal.xy *= scale;
  return normalize(tbn * tangent_normal);
}

float2 parallaxMapping(
    texture2d<float> height_map [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 2) uv [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) view_dir [[stage_in]],
    float height_scale [[stage_in]]) {
  float height =
      height_map.sample(sampler(mag_filter::linear, min_filter::linear), uv).r;
  float2 p = view_dir.xy / view_dir.z * height * height_scale;
  return uv - p;
}

float distributionGGX(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) N [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) H [[stage_in]],
    float roughness [[stage_in]]) {
  float a = roughness * roughness;
  float a2 = a * a;
  float NdotH = max(dot(N, H), 0.0);
  float NdotH2 = NdotH * NdotH;
  float num = a2;
  float denom = NdotH2 * a2 - 1.0 + 1.0;
  denom = PI * denom * denom;
  return num / max(denom, EPSILON);
}

float geometrySchlickGGX(float NdotV [[stage_in]],
                         float roughness [[stage_in]]) {
  float r = roughness + 1.0;
  float k = r * r / 8.0;
  float num = NdotV;
  float denom = NdotV * 1.0 - k + k;
  return num / max(denom, EPSILON);
}

float geometrySmith(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) N [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) V [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) L [[stage_in]],
    float roughness [[stage_in]]) {
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float ggx2 = geometrySchlickGGX(NdotV, roughness);
  float ggx1 = geometrySchlickGGX(NdotL, roughness);
  return ggx1 * ggx2;
}

float3 fresnelSchlick(float cosTheta [[stage_in]],
                      VectorType(element_type = PrimitiveType(name = float,
                                                              size_bits = None),
                                 size = 3) F0 [[stage_in]]) {
  return F0 + 1.0 - F0 * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float3 fresnelSchlickRoughness(
    float cosTheta [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) F0 [[stage_in]],
    float roughness [[stage_in]]) {
  return F0 + max(float3(1.0 - roughness), F0) -
         F0 * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float calculateShadow(texture2d<float> shadow_map [[stage_in]],
                      VectorType(element_type = PrimitiveType(name = float,
                                                              size_bits = None),
                                 size = 4) frag_pos_light_space [[stage_in]],
                      float bias [[stage_in]]) {
  float3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;
  proj_coords = proj_coords * 0.5 + 0.5;
  if (proj_coords.z > 1.0) {
  }
  float shadow = 0.0;
  float2 texel_size = 1.0 / textureSize(shadow_map, 0);
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

float3 calculateIBL(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) N [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) V [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) albedo [[stage_in]],
    float metallic [[stage_in]], float roughness [[stage_in]],
    EnvironmentData env [[stage_in]]) {
  float3 F0 = mix(float3(0.04), albedo, metallic);
  float3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
  float3 kS = F;
  float3 kD = 1.0 - kS;
  kD *= 1.0 - metallic;
  float3 irradiance =
      env.irradiance_map
          .sample(sampler(mag_filter::linear, min_filter::linear), N)
          .rgb;
  float3 diffuse = irradiance * albedo;
  float3 R = reflect(-V, N);
  float3 prefiltered_color =
      textureLod(env.prefilter_map, R, roughness * env.max_reflection_lod).rgb;
  float2 brdf = env.brdf_lut
                    .sample(sampler(mag_filter::linear, min_filter::linear),
                            float2(max(dot(N, V), 0.0), roughness))
                    .rg;
  float3 specular = prefiltered_color * F * brdf.x + brdf.y;
  return kD * diffuse + specular * env.exposure;
}

float3 calculateDirectLighting(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) N [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) V [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) L [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) albedo [[stage_in]],
    float metallic [[stage_in]], float roughness [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) light_color [[stage_in]]) {
  float3 H = normalize(V + L);
  float3 F0 = mix(float3(0.04), albedo, metallic);
  float NDF = distributionGGX(N, H, roughness);
  float G = geometrySmith(N, V, L, roughness);
  float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
  float3 kS = F;
  float3 kD = float3(1.0) - kS;
  kD *= 1.0 - metallic;
  float3 numerator = NDF * G * F;
  float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + EPSILON;
  float3 specular = numerator / denominator;
  float NdotL = max(dot(N, L), 0.0);
  return kD * albedo / PI + specular * light_color * NdotL;
}

float calculateAttenuation(
    LightData light [[stage_in]],
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) frag_pos [[stage_in]]) {
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

float3 reinhardToneMapping(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) color [[stage_in]]) {
  return color / color + float3(1.0);
}

float3 acesToneMapping(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) color [[stage_in]]) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp(color * a * color + b / color * c * color + d + e, 0.0, 1.0);
}

float3 gammaCorrection(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) color [[stage_in]],
    float gamma [[stage_in]]) {
  return pow(color, float3(1.0 / gamma));
}

VertexOutput main(VertexInput input [[stage_in]]) {
  VertexOutput output;
  float4 world_pos = model_matrix * float4(input.position, 1.0);
  output.world_position = world_pos.xyz;
  output.clip_position = camera.view_projection_matrix * world_pos;
  output.world_normal = normalize(normal_matrix * input.normal);
  output.world_tangent = normalize(normal_matrix * input.tangent);
  output.world_bitangent = cross(output.world_normal, output.world_tangent);
  output.tbn_matrix =
      mat3(output.world_tangent, output.world_bitangent, output.world_normal);
  output.uv = input.uv;
  output.color = input.color;
  if (settings.enable_shadows) {
    for (int i = 0;
         i < settings.shadow_cascade_count && i < MAX_SHADOW_CASCADES; ++i) {
      output.shadow_coords[i] = shadow_matrices[i] * world_pos;
    }
  }
  return output;
}

float4 main(FragmentInput input [[stage_in]]) {
  float2 uv = input.uv;
  if (settings.enable_parallax_mapping && material.has_height_map) {
    float3 view_dir = normalize(camera.position - input.world_position);
    float3 tangent_view_dir = transpose(input.tbn_matrix) * view_dir;
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
    N = getNormalFromMap(normal_map, uv, input.tbn_matrix,
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
    float3 radiance = light.color * light.intensity * attenuation;
    float shadow = 0.0;
    if (settings.enable_shadows && light.cast_shadows &&
        i < settings.shadow_cascade_count) {
      shadow = calculateShadow(shadow_maps[i], input.shadow_coords[i],
                               settings.shadow_bias);
    }
    Lo += calculateDirectLighting(N, V, L, albedo, metallic, roughness,
                                  radiance) *
              1.0 -
          shadow;
  }
  float3 ambient = float3(0.0);
  if (settings.enable_ibl) {
    ambient = calculateIBL(N, V, albedo, metallic, roughness, environment);
  } else {
    ambient = environment.ambient_color * albedo * ao;
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

// Vertex Shader
vertex void vertex_main() {}

// Fragment Shader
fragment void fragment_main(, texture2d<float> albedo_map [[texture(7)]],
                            texture2d<float> normal_map [[texture(8)]],
                            texture2d<float> metallic_roughness_map
                            [[texture(9)]],
                            texture2d<float> ao_map [[texture(10)]],
                            texture2d<float> emission_map [[texture(11)]],
                            texture2d<float> height_map [[texture(12)]],
                            texture2d<float> shadow_maps [[texture(13)]]) {}

// Compute Shader
kernel void kernel_precompute_environment() {
  int2 coord = ivec2(gl_GlobalInvocationID.xy);
  int2 size = imageSize(irradiance_map);
  if (coord.x >= size.x || coord.y >= size.y) {
  }
  float2 uv = float2(coord) + 0.5 / float2(size);
  uv = uv * 2.0 - 1.0;
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
      float3 sample_vec = tangent_sample.x * right + tangent_sample.y * up +
                          tangent_sample.z * N;
      irradiance += environment_map
                        .sample(sampler(mag_filter::linear, min_filter::linear),
                                sample_vec)
                        .rgb *
                    cos(theta) * sin(theta);
      ++sample_count;
    }
  }
  irradiance = PI * irradiance / sample_count;
  imageStore(irradiance_map, ivec3(coord, face_index), float4(irradiance, 1.0));
}

float3 getSamplingVector(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 2) uv [[stage_in]],
    int face [[stage_in]]) {
  float3 result;
  SwitchNode(cases = 6);
  return normalize(result);
}
