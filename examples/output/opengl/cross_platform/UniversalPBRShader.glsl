#version 450 core
const float PI = 3.14159265359;
const float EPSILON = 0.0001;
const int MAX_LIGHTS = 32;
const int MAX_SHADOW_CASCADES = 4;

#ifdef GL_VERTEX_SHADER
in vec3 position;
in vec3 normal;
in vec3 tangent;
in vec2 uv;
in vec4 color;
#endif
#ifdef GL_VERTEX_SHADER
out vec4 clip_position;
out vec3 world_position;
out vec3 world_normal;
out vec3 world_tangent;
out vec3 world_bitangent;
out vec2 out_uv;
out vec4 out_color;
out mat3 tbn_matrix;
out vec4 shadow_coords[MAX_SHADOW_CASCADES];
#endif
#ifdef GL_FRAGMENT_SHADER
in vec3 in_world_position;
in vec3 in_world_normal;
in vec3 in_world_tangent;
in vec3 in_world_bitangent;
in vec2 uv;
in vec4 color;
in mat3 in_tbn_matrix;
in vec4 in_shadow_coords[MAX_SHADOW_CASCADES];
#endif
struct MaterialProperties {
  vec3 albedo;
  float metallic;
  float roughness;
  float ao;
  vec3 emission;
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
  vec3 position;
  vec3 direction;
  vec3 color;
  float intensity;
  float range;
  float inner_cone_angle;
  float outer_cone_angle;
  int type;
  bool cast_shadows;
  mat4 light_view_proj;
};

struct EnvironmentData {
  samplerCube irradiance_map;
  samplerCube prefilter_map;
  sampler2D brdf_lut;
  float max_reflection_lod;
  float exposure;
  vec3 ambient_color;
};

struct CameraData {
  vec3 position;
  vec3 forward;
  vec3 up;
  vec3 right;
  mat4 view_matrix;
  mat4 projection_matrix;
  mat4 view_projection_matrix;
  float near_plane;
  float far_plane;
  float fov;
  vec2 screen_size;
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

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform mat3 normal_matrix;
uniform CameraData camera;
uniform RenderSettings settings;
uniform mat4 shadow_matrices[MAX_SHADOW_CASCADES];
layout(binding = 0) uniform sampler2D albedo_map;
layout(binding = 1) uniform sampler2D normal_map;
layout(binding = 2) uniform sampler2D metallic_roughness_map;
layout(binding = 3) uniform sampler2D ao_map;
layout(binding = 4) uniform sampler2D emission_map;
layout(binding = 5) uniform sampler2D height_map;
layout(binding = 6) uniform sampler2D shadow_maps[MAX_SHADOW_CASCADES];
uniform MaterialProperties material;
uniform EnvironmentData environment;
uniform LightData lights[MAX_LIGHTS];
uniform int active_light_count;
layout(binding = 10) uniform samplerCube environment_map;
layout(rgba16f, binding = 0) uniform imageCube irradiance_map;
uniform int face_index;
uniform int mip_level;
vec3 getNormalFromMap(sampler2D normal_map, vec2 uv, mat3 tbn, float scale) {
  vec3 tangent_normal = ((texture(normal_map, uv).xyz * 2.0) - 1.0);
  tangent_normal.xy *= scale;
  return normalize((tbn * tangent_normal));
}

vec2 parallaxMapping(sampler2D height_map, vec2 uv, vec3 view_dir,
                     float height_scale) {
  float height = texture(height_map, uv).r;
  vec2 p = ((view_dir.xy / view_dir.z) * (height * height_scale));
  return (uv - p);
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
  float a = (roughness * roughness);
  float a2 = (a * a);
  float NdotH = max(dot(N, H), 0.0);
  float NdotH2 = (NdotH * NdotH);
  float num = a2;
  float denom = ((NdotH2 * (a2 - 1.0)) + 1.0);
  denom = ((PI * denom) * denom);
  return (num / max(denom, EPSILON));
}

float geometrySchlickGGX(float NdotV, float roughness) {
  float r = (roughness + 1.0);
  float k = ((r * r) / 8.0);
  float num = NdotV;
  float denom = ((NdotV * (1.0 - k)) + k);
  return (num / max(denom, EPSILON));
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float ggx2 = geometrySchlickGGX(NdotV, roughness);
  float ggx1 = geometrySchlickGGX(NdotL, roughness);
  return (ggx1 * ggx2);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
  return (F0 + ((1.0 - F0) * pow(max((1.0 - cosTheta), 0.0), 5.0)));
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
  return (F0 + ((max(vec3((1.0 - roughness)), F0) - F0) *
                pow(max((1.0 - cosTheta), 0.0), 5.0)));
}

float calculateShadow(sampler2D shadow_map, vec4 frag_pos_light_space,
                      float bias) {
  vec3 proj_coords = (frag_pos_light_space.xyz / frag_pos_light_space.w);
  proj_coords = ((proj_coords * 0.5) + 0.5);
  if ((proj_coords.z > 1.0)) {
    return 0.0;
  }
  float shadow = 0.0;
  vec2 texel_size = (1.0 / vec2(textureSize(shadow_map, 0)));
  for (int x = (-1); (x <= 1); (++x)) {
    for (int y = (-1); (y <= 1); (++y)) {
      float pcf_depth =
          texture(shadow_map, (proj_coords.xy + (vec2(x, y) * texel_size))).r;
      shadow += (((proj_coords.z - bias) > pcf_depth) ? 1.0 : 0.0);
    }
  }
  return (shadow / 9.0);
}

vec3 calculateIBL(vec3 N, vec3 V, vec3 albedo, float metallic, float roughness,
                  EnvironmentData env) {
  vec3 F0 = mix(vec3(0.04), albedo, metallic);
  vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
  vec3 kS = F;
  vec3 kD = (1.0 - kS);
  kD *= (1.0 - metallic);
  vec3 irradiance = texture(env.irradiance_map, N).rgb;
  vec3 diffuse = (irradiance * albedo);
  vec3 R = reflect((-V), N);
  vec3 prefiltered_color =
      textureLod(env.prefilter_map, R, (roughness * env.max_reflection_lod))
          .rgb;
  vec2 brdf = texture(env.brdf_lut, vec2(max(dot(N, V), 0.0), roughness)).rg;
  vec3 specular = (prefiltered_color * ((F * brdf.x) + brdf.y));
  return (((kD * diffuse) + specular) * env.exposure);
}

vec3 calculateDirectLighting(vec3 N, vec3 V, vec3 L, vec3 albedo,
                             float metallic, float roughness,
                             vec3 light_color) {
  vec3 H = normalize((V + L));
  vec3 F0 = mix(vec3(0.04), albedo, metallic);
  float NDF = distributionGGX(N, H, roughness);
  float G = geometrySmith(N, V, L, roughness);
  vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
  vec3 kS = F;
  vec3 kD = (vec3(1.0) - kS);
  kD *= (1.0 - metallic);
  vec3 numerator = ((NDF * G) * F);
  float denominator =
      (((4.0 * max(dot(N, V), 0.0)) * max(dot(N, L), 0.0)) + EPSILON);
  vec3 specular = (numerator / denominator);
  float NdotL = max(dot(N, L), 0.0);
  return (((((kD * albedo) / PI) + specular) * light_color) * NdotL);
}

float calculateAttenuation(LightData light, vec3 frag_pos) {
  if ((light.type == 0)) {
    return 1.0;
  }
  float distance = length((light.position - frag_pos));
  if ((light.type == 1)) {
    float attenuation = (1.0 / (distance * distance));
    return (attenuation * smoothstep(light.range, 0.0, distance));
  }
  if ((light.type == 2)) {
    vec3 light_dir = normalize((light.position - frag_pos));
    float theta = dot(light_dir, normalize((-light.direction)));
    float epsilon = (light.inner_cone_angle - light.outer_cone_angle);
    float intensity =
        clamp(((theta - light.outer_cone_angle) / epsilon), 0.0, 1.0);
    float attenuation = (1.0 / (distance * distance));
    return ((attenuation * intensity) * smoothstep(light.range, 0.0, distance));
  }
  return 0.0;
}

vec3 reinhardToneMapping(vec3 color) { return (color / (color + vec3(1.0))); }

vec3 acesToneMapping(vec3 color) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp(
      ((color * ((a * color) + b)) / ((color * ((c * color) + d)) + e)), 0.0,
      1.0);
}

vec3 gammaCorrection(vec3 color, float gamma) {
  return pow(color, vec3((1.0 / gamma)));
}

#ifdef GL_VERTEX_SHADER
// Vertex Shader
void main() {
  vec4 world_pos = (model_matrix * vec4(position, 1.0));
  world_position = world_pos.xyz;
  clip_position = (camera.view_projection_matrix * world_pos);
  world_normal = normalize((normal_matrix * normal));
  world_tangent = normalize((normal_matrix * tangent));
  world_bitangent = cross(world_normal, world_tangent);
  tbn_matrix = mat3(world_tangent, world_bitangent, world_normal);
  out_uv = uv;
  out_color = color;
  if (settings.enable_shadows) {
    for (int i = 0;
         ((i < settings.shadow_cascade_count) && (i < MAX_SHADOW_CASCADES));
         (++i)) {
      shadow_coords[i] = (shadow_matrices[i] * world_pos);
    }
  }
  return;
}

#endif
#ifdef GL_FRAGMENT_SHADER
// Fragment Shader
layout(location = 0) out vec4 fragColor;
void main() {
  vec2 uv_ = uv;
  if ((settings.enable_parallax_mapping && material.has_height_map)) {
    vec3 view_dir = normalize((camera.position - in_world_position));
    vec3 tangent_view_dir = (transpose(in_tbn_matrix) * view_dir);
    uv_ = parallaxMapping(height_map, uv_, tangent_view_dir,
                          material.height_scale);
  }
  vec3 albedo = material.albedo;
  if (material.has_albedo_map) {
    albedo *= texture(albedo_map, uv_).rgb;
  }
  albedo *= color.rgb;
  float metallic = material.metallic;
  float roughness = material.roughness;
  if (material.has_metallic_roughness_map) {
    vec3 mr_sample = texture(metallic_roughness_map, uv_).rgb;
    metallic *= mr_sample.b;
    roughness *= mr_sample.g;
  }
  float ao = material.ao;
  if (material.has_ao_map) {
    ao *= texture(ao_map, uv_).r;
  }
  vec3 emission = material.emission;
  if (material.has_emission_map) {
    emission *= texture(emission_map, uv_).rgb;
  }
  vec3 N = normalize(in_world_normal);
  if ((settings.enable_normal_mapping && material.has_normal_map)) {
    N = getNormalFromMap(normal_map, uv_, in_tbn_matrix, material.normal_scale);
  }
  vec3 V = normalize((camera.position - in_world_position));
  vec3 Lo = vec3(0.0);
  for (int i = 0; ((i < active_light_count) && (i < MAX_LIGHTS)); (++i)) {
    LightData light = lights[i];
    vec3 L;
    if ((light.type == 0)) {
      L = normalize((-light.direction));
    } else {
      L = normalize((light.position - in_world_position));
    }
    float attenuation = calculateAttenuation(light, in_world_position);
    vec3 radiance = ((light.color * light.intensity) * attenuation);
    float shadow = 0.0;
    if (((settings.enable_shadows && light.cast_shadows) &&
         (i < settings.shadow_cascade_count))) {
      shadow = calculateShadow(shadow_maps[i], in_shadow_coords[i],
                               settings.shadow_bias);
    }
    Lo += (calculateDirectLighting(N, V, L, albedo, metallic, roughness,
                                   radiance) *
           (1.0 - shadow));
  }
  vec3 ambient = vec3(0.0);
  if (settings.enable_ibl) {
    ambient = calculateIBL(N, V, albedo, metallic, roughness, environment);
  } else {
    ambient = ((environment.ambient_color * albedo) * ao);
  }
  vec3 color_ = ((ambient + Lo) + emission);
  if (settings.enable_tone_mapping) {
    color_ = acesToneMapping(color_);
  }
  if (settings.enable_gamma_correction) {
    color_ = gammaCorrection(color_, 2.2);
  }
  fragColor = vec4(color_, color.a);
  return;
}

#endif
#ifdef GL_COMPUTE_SHADER
// Compute Shader
vec3 getSamplingVector(vec2 uv, int face) {
  vec3 result;
  switch (face) {
    case 0: {
      result = vec3(1.0, (-uv.y), (-uv.x));
      break;
    }
    case 1: {
      result = vec3((-1.0), (-uv.y), uv.x);
      break;
    }
    case 2: {
      result = vec3(uv.x, 1.0, uv.y);
      break;
    }
    case 3: {
      result = vec3(uv.x, (-1.0), (-uv.y));
      break;
    }
    case 4: {
      result = vec3(uv.x, (-uv.y), 1.0);
      break;
    }
    case 5: {
      result = vec3((-uv.x), (-uv.y), (-1.0));
      break;
    }
  }
  return normalize(result);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size = imageSize(irradiance_map);
  if (((coord.x >= size.x) || (coord.y >= size.y))) {
    return;
  }
  vec2 uv = ((vec2(coord) + 0.5) / vec2(size));
  uv = ((uv * 2.0) - 1.0);
  vec3 N = getSamplingVector(uv, face_index);
  vec3 irradiance = vec3(0.0);
  float sample_count = 0.0;
  for (float phi = 0.0; (phi < (2.0 * PI)); phi += 0.025) {
    for (float theta = 0.0; (theta < (0.5 * PI)); theta += 0.025) {
      vec3 tangent_sample =
          vec3((sin(theta) * cos(phi)), (sin(theta) * sin(phi)), cos(theta));
      vec3 up =
          ((abs(N.z) < 0.999) ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0));
      vec3 right = normalize(cross(up, N));
      up = normalize(cross(N, right));
      vec3 sample_vec =
          (((tangent_sample.x * right) + (tangent_sample.y * up)) +
           (tangent_sample.z * N));
      irradiance += ((texture(environment_map, sample_vec).rgb * cos(theta)) *
                     sin(theta));
      (++sample_count);
    }
  }
  irradiance = ((PI * irradiance) / sample_count);
  imageStore(irradiance_map, ivec3(coord, face_index), vec4(irradiance, 1.0));
}

#endif
