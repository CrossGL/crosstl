
#version 450 core
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
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) light_view_proj;
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
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) view_matrix;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) projection_matrix;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) view_projection_matrix;
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
struct VertexOutput {
  vec4 clip_position;
  vec3 world_position;
  vec3 world_normal;
  vec3 world_tangent;
  vec3 world_bitangent;
  vec2 uv;
  vec4 color;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 3, cols = 3) tbn_matrix;
  vec4 shadow_coords[MAX_SHADOW_CASCADES];
};
layout(std140, binding = 0)
    MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
               rows = 4, cols = 4) model_matrix;
layout(std140, binding = 1)
    MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
               rows = 4, cols = 4) view_matrix;
layout(std140, binding = 2)
    MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
               rows = 4, cols = 4) projection_matrix;
layout(std140, binding = 3)
    MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
               rows = 3, cols = 3) normal_matrix;
layout(std140, binding = 4) CameraData camera;
layout(std140, binding = 5) RenderSettings settings;
layout(std140, binding = 6)
    MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
               rows = 4, cols = 4)[] shadow_matrices;
layout(std140, binding = 7) sampler2D albedo_map;
layout(std140, binding = 8) sampler2D normal_map;
layout(std140, binding = 9) sampler2D metallic_roughness_map;
layout(std140, binding = 10) sampler2D ao_map;
layout(std140, binding = 11) sampler2D emission_map;
layout(std140, binding = 12) sampler2D height_map;
layout(std140, binding = 13) sampler2D[] shadow_maps;
layout(std140, binding = 14) MaterialProperties material;
layout(std140, binding = 15) EnvironmentData environment;
layout(std140, binding = 16) CameraData camera;
layout(std140, binding = 17) RenderSettings settings;
layout(std140, binding = 18) LightData[] lights;
layout(std140, binding = 19) int active_light_count;
VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    getNormalFromMap(
        sampler2D normal_map,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 2) uv,
        MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
                   rows = 3, cols = 3) tbn,
        float scale) {
  vec3 tangent_normal =
      ((IdentifierNode(name = texture)(normal_map, uv).xyz * 2.0) - 1.0);
  tangent_normal.xy *= scale;
  return IdentifierNode(name = normalize)((tbn * tangent_normal));
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 2)
    parallaxMapping(
        sampler2D height_map,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 2) uv,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) view_dir,
        float height_scale) {
  float height = IdentifierNode(name = texture)(height_map, uv).r;
  vec2 p = ((view_dir.xy / view_dir.z) * (height * height_scale));
  return (uv - p);
}

float distributionGGX(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) N,
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) H,
    float roughness) {
  float a = (roughness * roughness);
  float a2 = (a * a);
  float NdotH =
      IdentifierNode(name = max)(IdentifierNode(name = dot)(N, H), 0.0);
  float NdotH2 = (NdotH * NdotH);
  float num = a2;
  float denom = ((NdotH2 * (a2 - 1.0)) + 1.0);
  denom = ((PI * denom) * denom);
  return (num / IdentifierNode(name = max)(denom, EPSILON));
}

float geometrySchlickGGX(float NdotV, float roughness) {
  float r = (roughness + 1.0);
  float k = ((r * r) / 8.0);
  float num = NdotV;
  float denom = ((NdotV * (1.0 - k)) + k);
  return (num / IdentifierNode(name = max)(denom, EPSILON));
}

float geometrySmith(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) N,
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) V,
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) L,
    float roughness) {
  float NdotV =
      IdentifierNode(name = max)(IdentifierNode(name = dot)(N, V), 0.0);
  float NdotL =
      IdentifierNode(name = max)(IdentifierNode(name = dot)(N, L), 0.0);
  float ggx2 = IdentifierNode(name = geometrySchlickGGX)(NdotV, roughness);
  float ggx1 = IdentifierNode(name = geometrySchlickGGX)(NdotL, roughness);
  return (ggx1 * ggx2);
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    fresnelSchlick(float cosTheta,
                   VectorType(element_type = PrimitiveType(name = float,
                                                           size_bits = None),
                              size = 3) F0) {
  return (F0 + ((1.0 - F0) *
                IdentifierNode(name = pow)(
                    IdentifierNode(name = max)((1.0 - cosTheta), 0.0), 5.0)));
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    fresnelSchlickRoughness(
        float cosTheta,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) F0,
        float roughness) {
  return (F0 + ((IdentifierNode(name = max)(
                     IdentifierNode(name = vec3)((1.0 - roughness)), F0) -
                 F0) *
                IdentifierNode(name = pow)(
                    IdentifierNode(name = max)((1.0 - cosTheta), 0.0), 5.0)));
}

float calculateShadow(sampler2D shadow_map,
                      VectorType(element_type = PrimitiveType(name = float,
                                                              size_bits = None),
                                 size = 4) frag_pos_light_space,
                      float bias) {
  vec3 proj_coords = (frag_pos_light_space.xyz / frag_pos_light_space.w);
  proj_coords = ((proj_coords * 0.5) + 0.5);
  if ((proj_coords.z > 1.0)) {
  }
  float shadow = 0.0;
  vec2 texel_size = (1.0 / IdentifierNode(name = textureSize)(shadow_map, 0));
  for (int x = (-1);; (x <= 1); (++x)) {
    for (int y = (-1);; (y <= 1); (++y)) {
      float pcf_depth =
          IdentifierNode(name = texture)(
              shadow_map, (proj_coords.xy +
                           (IdentifierNode(name = vec2)(x, y) * texel_size)))
              .r;
      shadow += (((proj_coords.z - bias) > pcf_depth) ? 1.0 : 0.0);
    }
  }
  return (shadow / 9.0);
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    calculateIBL(
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) N,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) V,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) albedo,
        float metallic, float roughness, EnvironmentData env) {
  vec3 F0 = IdentifierNode(name = mix)(IdentifierNode(name = vec3)(0.04),
                                       albedo, metallic);
  vec3 F = IdentifierNode(name = fresnelSchlickRoughness)(
      IdentifierNode(name = max)(IdentifierNode(name = dot)(N, V), 0.0), F0,
      roughness);
  vec3 kS = F;
  vec3 kD = (1.0 - kS);
  kD *= (1.0 - metallic);
  vec3 irradiance = IdentifierNode(name = texture)(env.irradiance_map, N).rgb;
  vec3 diffuse = (irradiance * albedo);
  vec3 R = IdentifierNode(name = reflect)((-V), N);
  vec3 prefiltered_color =
      IdentifierNode(name = textureLod)(env.prefilter_map, R,
                                        (roughness * env.max_reflection_lod))
          .rgb;
  vec2 brdf =
      IdentifierNode(name = texture)(
          env.brdf_lut,
          IdentifierNode(name = vec2)(
              IdentifierNode(name = max)(IdentifierNode(name = dot)(N, V), 0.0),
              roughness))
          .rg;
  vec3 specular = (prefiltered_color * ((F * brdf.x) + brdf.y));
  return (((kD * diffuse) + specular) * env.exposure);
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    calculateDirectLighting(
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) N,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) V,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) L,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) albedo,
        float metallic, float roughness,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) light_color) {
  vec3 H = IdentifierNode(name = normalize)((V + L));
  vec3 F0 = IdentifierNode(name = mix)(IdentifierNode(name = vec3)(0.04),
                                       albedo, metallic);
  float NDF = IdentifierNode(name = distributionGGX)(N, H, roughness);
  float G = IdentifierNode(name = geometrySmith)(N, V, L, roughness);
  vec3 F = IdentifierNode(name = fresnelSchlick)(
      IdentifierNode(name = max)(IdentifierNode(name = dot)(H, V), 0.0), F0);
  vec3 kS = F;
  vec3 kD = (IdentifierNode(name = vec3)(1.0) - kS);
  kD *= (1.0 - metallic);
  vec3 numerator = ((NDF * G) * F);
  float denominator =
      (((4.0 *
         IdentifierNode(name = max)(IdentifierNode(name = dot)(N, V), 0.0)) *
        IdentifierNode(name = max)(IdentifierNode(name = dot)(N, L), 0.0)) +
       EPSILON);
  vec3 specular = (numerator / denominator);
  float NdotL =
      IdentifierNode(name = max)(IdentifierNode(name = dot)(N, L), 0.0);
  return (((((kD * albedo) / PI) + specular) * light_color) * NdotL);
}

float calculateAttenuation(
    LightData light,
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) frag_pos) {
  if ((light.type == 0)) {
    return 1.0;
  }
  float distance = IdentifierNode(name = length)((light.position - frag_pos));
  if ((light.type == 1)) {
    float attenuation = (1.0 / (distance * distance));
    return (attenuation *
            IdentifierNode(name = smoothstep)(light.range, 0.0, distance));
  }
  if ((light.type == 2)) {
    vec3 light_dir =
        IdentifierNode(name = normalize)((light.position - frag_pos));
    float theta = IdentifierNode(name = dot)(
        light_dir, IdentifierNode(name = normalize)((-light.direction)));
    float epsilon = (light.inner_cone_angle - light.outer_cone_angle);
    float intensity = IdentifierNode(name = clamp)(
        ((theta - light.outer_cone_angle) / epsilon), 0.0, 1.0);
    float attenuation = (1.0 / (distance * distance));
    return ((attenuation * intensity) *
            IdentifierNode(name = smoothstep)(light.range, 0.0, distance));
  }
  return 0.0;
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    reinhardToneMapping(
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) color) {
  return (color / (color + IdentifierNode(name = vec3)(1.0)));
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    acesToneMapping(VectorType(element_type = PrimitiveType(name = float,
                                                            size_bits = None),
                               size = 3) color) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return IdentifierNode(name = clamp)(
      ((color * ((a * color) + b)) / ((color * ((c * color) + d)) + e)), 0.0,
      1.0);
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    gammaCorrection(VectorType(element_type = PrimitiveType(name = float,
                                                            size_bits = None),
                               size = 3) color,
                    float gamma) {
  return IdentifierNode(name = pow)(color,
                                    IdentifierNode(name = vec3)((1.0 / gamma)));
}

VertexOutput main(VertexInput input) {
  VertexOutput output;
  vec4 world_pos =
      (model_matrix * IdentifierNode(name = vec4)(input.position, 1.0));
  output.world_position = world_pos.xyz;
  output.clip_position = (camera.view_projection_matrix * world_pos);
  output.world_normal =
      IdentifierNode(name = normalize)((normal_matrix * input.normal));
  output.world_tangent =
      IdentifierNode(name = normalize)((normal_matrix * input.tangent));
  output.world_bitangent =
      IdentifierNode(name = cross)(output.world_normal, output.world_tangent);
  output.tbn_matrix = IdentifierNode(name = mat3)(
      output.world_tangent, output.world_bitangent, output.world_normal);
  output.uv = input.uv;
  output.color = input.color;
  if (settings.enable_shadows) {
    for (int i = 0;;
         ((i < settings.shadow_cascade_count) && (i < MAX_SHADOW_CASCADES));
         (++i)) {
      output.shadow_coords[i] = (shadow_matrices[i] * world_pos);
    }
  }
  return output;
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 4) main(FragmentInput input) {
  vec2 uv = input.uv;
  if ((settings.enable_parallax_mapping && material.has_height_map)) {
    vec3 view_dir = IdentifierNode(name = normalize)(
        (camera.position - input.world_position));
    vec3 tangent_view_dir =
        (IdentifierNode(name = transpose)(input.tbn_matrix) * view_dir);
    uv = IdentifierNode(name = parallaxMapping)(
        height_map, uv, tangent_view_dir, material.height_scale);
  }
  vec3 albedo = material.albedo;
  if (material.has_albedo_map) {
    albedo *= IdentifierNode(name = texture)(albedo_map, uv).rgb;
  }
  albedo *= input.color.rgb;
  float metallic = material.metallic;
  float roughness = material.roughness;
  if (material.has_metallic_roughness_map) {
    vec3 mr_sample =
        IdentifierNode(name = texture)(metallic_roughness_map, uv).rgb;
    metallic *= mr_sample.b;
    roughness *= mr_sample.g;
  }
  float ao = material.ao;
  if (material.has_ao_map) {
    ao *= IdentifierNode(name = texture)(ao_map, uv).r;
  }
  vec3 emission = material.emission;
  if (material.has_emission_map) {
    emission *= IdentifierNode(name = texture)(emission_map, uv).rgb;
  }
  vec3 N = IdentifierNode(name = normalize)(input.world_normal);
  if ((settings.enable_normal_mapping && material.has_normal_map)) {
    N = IdentifierNode(name = getNormalFromMap)(
        normal_map, uv, input.tbn_matrix, material.normal_scale);
  }
  vec3 V = IdentifierNode(name = normalize)(
      (camera.position - input.world_position));
  vec3 Lo = IdentifierNode(name = vec3)(0.0);
  for (int i = 0;; ((i < active_light_count) && (i < MAX_LIGHTS)); (++i)) {
    LightData light = lights[i];
    vec3 L;
    if ((light.type == 0)) {
      L = IdentifierNode(name = normalize)((-light.direction));
    } else {
      L = IdentifierNode(name = normalize)(
          (light.position - input.world_position));
    }
    float attenuation = IdentifierNode(name = calculateAttenuation)(
        light, input.world_position);
    vec3 radiance = ((light.color * light.intensity) * attenuation);
    float shadow = 0.0;
    if (((settings.enable_shadows && light.cast_shadows) &&
         (i < settings.shadow_cascade_count))) {
      shadow = IdentifierNode(name = calculateShadow)(
          shadow_maps[i], input.shadow_coords[i], settings.shadow_bias);
    }
    Lo += (IdentifierNode(name = calculateDirectLighting)(
               N, V, L, albedo, metallic, roughness, radiance) *
           (1.0 - shadow));
  }
  vec3 ambient = IdentifierNode(name = vec3)(0.0);
  if (settings.enable_ibl) {
    ambient = IdentifierNode(name = calculateIBL)(N, V, albedo, metallic,
                                                  roughness, environment);
  } else {
    ambient = ((environment.ambient_color * albedo) * ao);
  }
  vec3 color = ((ambient + Lo) + emission);
  if (settings.enable_tone_mapping) {
    color = IdentifierNode(name = acesToneMapping)(color);
  }
  if (settings.enable_gamma_correction) {
    color = IdentifierNode(name = gammaCorrection)(color, 2.2);
  }
  return IdentifierNode(name = vec4)(color, input.color.a);
}

// Vertex Shader
void main() {}

// Fragment Shader
void main() {}

// Compute Shader
void main() {
  ivec2 coord = IdentifierNode(name = ivec2)(gl_GlobalInvocationID.xy);
  ivec2 size = IdentifierNode(name = imageSize)(irradiance_map);
  if (((coord.x >= size.x) || (coord.y >= size.y))) {
  }
  vec2 uv = ((IdentifierNode(name = vec2)(coord) + 0.5) /
             IdentifierNode(name = vec2)(size));
  uv = ((uv * 2.0) - 1.0);
  vec3 N = IdentifierNode(name = getSamplingVector)(uv, face_index);
  vec3 irradiance = IdentifierNode(name = vec3)(0.0);
  float sample_count = 0.0;
  for (float phi = 0.0;; (phi < (2.0 * PI)); phi += 0.025) {
    for (float theta = 0.0;; (theta < (0.5 * PI)); theta += 0.025) {
      vec3 tangent_sample = IdentifierNode(name = vec3)(
          (IdentifierNode(name = sin)(theta) * IdentifierNode(name = cos)(phi)),
          (IdentifierNode(name = sin)(theta) * IdentifierNode(name = sin)(phi)),
          IdentifierNode(name = cos)(theta));
      vec3 up = ((IdentifierNode(name = abs)(N.z) < 0.999)
                     ? IdentifierNode(name = vec3)(0.0, 0.0, 1.0)
                     : IdentifierNode(name = vec3)(1.0, 0.0, 0.0));
      vec3 right =
          IdentifierNode(name = normalize)(IdentifierNode(name = cross)(up, N));
      up = IdentifierNode(name = normalize)(
          IdentifierNode(name = cross)(N, right));
      vec3 sample_vec =
          (((tangent_sample.x * right) + (tangent_sample.y * up)) +
           (tangent_sample.z * N));
      irradiance +=
          ((IdentifierNode(name = texture)(environment_map, sample_vec).rgb *
            IdentifierNode(name = cos)(theta)) *
           IdentifierNode(name = sin)(theta));
      (++sample_count);
    }
  }
  irradiance = ((PI * irradiance) / sample_count);
  IdentifierNode(name = imageStore)(
      irradiance_map, IdentifierNode(name = ivec3)(coord, face_index),
      IdentifierNode(name = vec4)(irradiance, 1.0));
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    getSamplingVector(VectorType(element_type = PrimitiveType(name = float,
                                                              size_bits = None),
                                 size = 2) uv,
                      int face) {
  vec3 result;
  SwitchNode(cases = 6);
  return IdentifierNode(name = normalize)(result);
}
