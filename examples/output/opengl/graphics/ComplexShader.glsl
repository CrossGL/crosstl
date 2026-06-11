#version 450 core
const float PI = 3.14159265359;
const float EPSILON = 0.0001;
const int MAX_ITERATIONS = 64;
const vec3 UP_VECTOR = vec3(0.0, 1.0, 0.0);

#ifdef GL_VERTEX_SHADER
in vec3 position;
in vec3 normal;
in vec3 tangent;
in vec3 bitangent;
in vec2 texCoord0;
in vec2 texCoord1;
in vec4 color;
in int materialIndex;
#endif
#ifdef GL_VERTEX_SHADER
out vec3 worldPosition;
out vec3 worldNormal;
out vec3 worldTangent;
out vec3 worldBitangent;
out vec2 out_texCoord0;
out vec2 out_texCoord1;
out vec4 out_color;
out mat3 TBN;
flat out int out_materialIndex;
out vec4 clipPosition;
#endif
#ifdef GL_FRAGMENT_SHADER
in vec3 in_worldPosition;
in vec3 in_worldNormal;
in vec3 in_worldTangent;
in vec3 in_worldBitangent;
in vec2 in_out_texCoord0;
in vec2 in_out_texCoord1;
in vec4 in_out_color;
in mat3 in_TBN;
flat in int in_out_materialIndex;
in vec4 in_clipPosition;
#endif
#ifdef GL_FRAGMENT_SHADER
layout(location = 0) out vec4 out_color_2;
layout(location = 1) out vec4 normalBuffer;
layout(location = 2) out vec4 positionBuffer;
layout(location = 3) out float depth;
#endif
struct Material {
  vec3 albedo;
  float roughness;
  float metallic;
  vec3 emissive;
  float opacity;
  bool hasNormalMap;
  sampler2D albedoMap;
  sampler2D normalMap;
  sampler2D metallicRoughnessMap;
};

struct Light {
  vec3 position;
  vec3 color;
  float intensity;
  float radius;
  bool castShadows;
  mat4 viewProjection;
};

struct Scene {
  Material materials[4];
  Light lights[8];
  vec3 ambientLight;
  float time;
  float elapsedTime;
  int activeLightCount;
  mat4 viewMatrix;
  mat4 projectionMatrix;
};

struct FragmentOutput {
  vec4 color;
  vec4 normalBuffer;
  vec4 positionBuffer;
  float depth;
};

struct GlobalUniforms {
  Scene scene;
  vec3 cameraPosition;
  float globalRoughness;
  vec2 screenSize;
  float nearPlane;
  float farPlane;
  int frameCount;
  float noiseValues[1024];
};

uniform GlobalUniforms globals;
layout(binding = 0) uniform sampler2D shadowMap;
layout(rgba8, binding = 0) uniform image2D outputImage;
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

float noise3D(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = (((f * f) * f) * ((f * ((f * 6.0) - 15.0)) + 10.0));
  float n000 = fract((sin(dot(i, vec3(13.534, 43.5234, 243.32))) * 4453.0));
  float n001 = fract(
      (sin(dot((i + vec3(0.0, 0.0, 1.0)), vec3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n010 = fract(
      (sin(dot((i + vec3(0.0, 1.0, 0.0)), vec3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n011 = fract(
      (sin(dot((i + vec3(0.0, 1.0, 1.0)), vec3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n100 = fract(
      (sin(dot((i + vec3(1.0, 0.0, 0.0)), vec3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n101 = fract(
      (sin(dot((i + vec3(1.0, 0.0, 1.0)), vec3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n110 = fract(
      (sin(dot((i + vec3(1.0, 1.0, 0.0)), vec3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n111 = fract(
      (sin(dot((i + vec3(1.0, 1.0, 1.0)), vec3(13.534, 43.5234, 243.32))) *
       4453.0));
  float n00 = mix(n000, n001, u.z);
  float n01 = mix(n010, n011, u.z);
  float n10 = mix(n100, n101, u.z);
  float n11 = mix(n110, n111, u.z);
  float n0 = mix(n00, n01, u.y);
  float n1 = mix(n10, n11, u.y);
  return mix(n0, n1, u.x);
}

float fbm(vec3 p, int octaves, float lacunarity, float gain) {
  float sum = 0.0;
  float amplitude = 1.0;
  float frequency = 1.0;
  for (int i = 0; (i < octaves); (++i)) {
    if ((i >= MAX_ITERATIONS)) {
      break;
    }
    sum += (amplitude * noise3D((p * frequency)));
    amplitude *= gain;
    frequency *= lacunarity;
  }
  return sum;
}

vec4 samplePlanarProjection(sampler2D tex, vec3 worldPos, vec3 normal) {
  vec3 absNormal = abs(normal);
  bool useX = ((absNormal.x >= absNormal.y) && (absNormal.x >= absNormal.z));
  bool useY = ((!useX) && (absNormal.y >= absNormal.z));
  vec2 uv;
  if (useX) {
    uv = ((worldPos.zy * 0.5) + 0.5);
    if ((normal.x < 0.0)) {
      uv.x = (1.0 - uv.x);
    }
  } else {
    if (useY) {
      uv = ((worldPos.xz * 0.5) + 0.5);
      if ((normal.y < 0.0)) {
        uv.y = (1.0 - uv.y);
      }
    } else {
      uv = ((worldPos.xy * 0.5) + 0.5);
      if ((normal.z < 0.0)) {
        uv.x = (1.0 - uv.x);
      }
    }
  }
  return texture(tex, uv);
}

#ifdef GL_VERTEX_SHADER
// Vertex Shader
void main() {
  mat4 modelMatrix = mat4(1.0);
  mat4 viewMatrix = globals.scene.viewMatrix;
  mat4 projectionMatrix = globals.scene.projectionMatrix;
  mat4 modelViewMatrix = (viewMatrix * modelMatrix);
  mat4 modelViewProjectionMatrix = (projectionMatrix * modelViewMatrix);
  mat3 normalMatrix = mat3(transpose(inverse(modelMatrix)));
  vec4 worldPosition_ = (modelMatrix * vec4(position, 1.0));
  vec3 worldNormal_ = normalize((normalMatrix * normal));
  vec3 worldTangent_ = normalize((normalMatrix * tangent));
  vec3 worldBitangent_ = normalize((normalMatrix * bitangent));
  mat3 TBN_ = mat3(worldTangent_, worldBitangent_, worldNormal_);
  float displacement =
      (fbm((worldPosition_.xyz + (globals.scene.time * 0.1)), 4, 2.0, 0.5) *
       0.1);
  if ((materialIndex > 0)) {
    worldPosition_.xyz += (worldNormal_ * displacement);
  }
  vec3 viewDir = normalize((globals.cameraPosition - worldPosition_.xyz));
  float fresnel = pow((1.0 - max(0.0, dot(worldNormal_, viewDir))), 5.0);
  if ((materialIndex < globals.scene.activeLightCount)) {
    out_color = (color * vec4(1.0, 1.0, 1.0, 1.0));
    for (int i = 0; (i < 4); (++i)) {
      if ((i >= (globals.frameCount % 5))) {
        break;
      }
      Light light = globals.scene.lights[i];
      vec3 lightDir = normalize((light.position - worldPosition_.xyz));
      float lightDistance = length((light.position - worldPosition_.xyz));
      float attenuation = (1.0 / (1.0 + (lightDistance * lightDistance)));
      float lightIntensity = (light.intensity * attenuation);
      out_color.rgb += (((light.color * lightIntensity) *
                         max(0.0, dot(worldNormal_, lightDir))) *
                        0.025);
    }
  } else {
    out_color = color;
    if ((globals.globalRoughness > 0.5)) {
      if ((fresnel > 0.7)) {
        out_color.a *= 0.8;
      } else {
        out_color.a *= 0.9;
      }
    }
  }
  worldPosition = worldPosition_.xyz;
  worldNormal = worldNormal_;
  worldTangent = worldTangent_;
  worldBitangent = worldBitangent_;
  out_texCoord0 = texCoord0;
  out_texCoord1 = texCoord1;
  TBN = TBN_;
  out_materialIndex = materialIndex;
  clipPosition = (modelViewProjectionMatrix * vec4(position, 1.0));
  return;
}

#endif
#ifdef GL_FRAGMENT_SHADER
// Fragment Shader
float shadowCalculation(vec4 fragPosLightSpace, int iteration);

float shadowCalculation(vec4 fragPosLightSpace, int iteration) {
  if ((iteration > 3)) {
    return 0.0;
  }
  vec3 projCoords = (fragPosLightSpace.xyz / fragPosLightSpace.w);
  projCoords = ((projCoords * 0.5) + 0.5);
  float closestDepth = texture(shadowMap, projCoords.xy).r;
  float currentDepth = projCoords.z;
  float bias =
      max((0.05 * (1.0 - dot(in_worldNormal, normalize((globals.cameraPosition -
                                                        in_worldPosition))))),
          0.005);
  float shadow = (((currentDepth - bias) > closestDepth) ? 1.0 : 0.0);
  float pcfDepth = 0.0;
  vec2 texelSize = (1.0 / vec2(globals.screenSize));
  float offset = (globals.noiseValues[((iteration * 4) % 16)] * 0.001);
  for (int x = (-1); (x <= 1); (++x)) {
    for (int y = (-1); (y <= 1); (++y)) {
      float pcfDepth =
          texture(shadowMap,
                  ((projCoords.xy + (vec2(x, y) * texelSize)) + vec2(offset)))
              .r;
      shadow += (((currentDepth - bias) > pcfDepth) ? 1.0 : 0.0);
    }
  }
  shadow /= 9.0;
  if ((projCoords.z > 1.0)) {
    shadow = 0.0;
  }
  return shadow;
}

void main() {
  vec4 albedoValue =
      texture(globals.scene.materials[in_out_materialIndex].albedoMap,
              in_out_texCoord0);
  vec4 normalValue =
      texture(globals.scene.materials[in_out_materialIndex].normalMap,
              in_out_texCoord0);
  vec4 metallicRoughnessValue = texture(
      globals.scene.materials[in_out_materialIndex].metallicRoughnessMap,
      in_out_texCoord0);
  vec3 normal = ((normalValue.xyz * 2.0) - 1.0);
  vec3 worldNormal = normalize((in_TBN * normal));
  vec3 albedo =
      (albedoValue.rgb * globals.scene.materials[in_out_materialIndex].albedo);
  float metallic = (metallicRoughnessValue.b *
                    globals.scene.materials[in_out_materialIndex].metallic);
  float roughness = (metallicRoughnessValue.g *
                     globals.scene.materials[in_out_materialIndex].roughness);
  float ao = metallicRoughnessValue.r;
  vec3 viewDir = normalize((globals.cameraPosition - in_worldPosition));
  vec3 F0 = mix(vec3(0.04), albedo, metallic);
  vec3 Lo = vec3(0.0);
  for (int i = 0; (i < globals.scene.activeLightCount); (++i)) {
    if ((i >= 8)) {
      break;
    }
    Light light = globals.scene.lights[i];
    vec3 lightDir = normalize((light.position - in_worldPosition));
    vec3 halfway = normalize((viewDir + lightDir));
    float distance = length((light.position - in_worldPosition));
    float attenuation = (1.0 / (distance * distance));
    vec3 radiance = ((light.color * light.intensity) * attenuation);
    float NDF = distributionGGX(worldNormal, halfway, roughness);
    float G = geometrySmith(worldNormal, viewDir, lightDir, roughness);
    vec3 F = fresnelSchlick(max(dot(halfway, viewDir), 0.0), F0);
    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS);
    kD *= (1.0 - metallic);
    vec3 numerator = ((NDF * G) * F);
    float denominator = (((4.0 * max(dot(worldNormal, viewDir), 0.0)) *
                          max(dot(worldNormal, lightDir), 0.0)) +
                         EPSILON);
    vec3 specular = (numerator / denominator);
    float NdotL = max(dot(worldNormal, lightDir), 0.0);
    float shadow = 0.0;
    if (light.castShadows) {
      vec4 fragPosLightSpace =
          (light.viewProjection * vec4(in_worldPosition, 1.0));
      shadow = shadowCalculation(fragPosLightSpace, 0);
      for (int s = 0; (s < 4); (++s)) {
        if ((s >= (globals.frameCount % 3))) {
          continue;
        }
        shadow += shadowCalculation(
            (fragPosLightSpace +
             vec4((globals.noiseValues[(s % 16)] * 0.001), 0.0, 0.0, 0.0)),
            (s + 1));
      }
      shadow /= 5.0;
    }
    Lo += ((((1.0 - shadow) * (((kD * albedo) / PI) + specular)) * radiance) *
           NdotL);
  }
  vec3 ambient = ((globals.scene.ambientLight * albedo) * ao);
  vec3 color = (ambient + Lo);
  color = (color / (color + vec3(1.0)));
  color = pow(color, vec3((1.0 / 2.2)));
  out_color_2 = vec4(
      color,
      (globals.scene.materials[in_out_materialIndex].opacity * albedoValue.a));
  normalBuffer = vec4(((worldNormal * 0.5) + 0.5), 1.0);
  positionBuffer = vec4(in_worldPosition, 1.0);
  depth = (in_clipPosition.z / in_clipPosition.w);
  return;
}

#endif
#ifdef GL_COMPUTE_SHADER
// Compute Shader
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main() {
  ivec2 texCoord = ivec2(gl_GlobalInvocationID.xy);
  vec2 screenSize = globals.screenSize;
  if (((texCoord.x >= int(screenSize.x)) ||
       (texCoord.y >= int(screenSize.y)))) {
    return;
  }
  vec2 uv = (vec2(texCoord) / screenSize);
  vec4 color = vec4(0.0);
  float totalWeight = 0.0;
  vec2 direction = (vec2(0.5) - uv);
  float len = length(direction);
  direction = normalize(direction);
  for (int i = 0; (i < 32); (++i)) {
    if ((i >= MAX_ITERATIONS)) {
      break;
    }
    float t = (float(i) / 32.0);
    vec2 pos = (uv + (((direction * t) * len) * 0.1));
    float noise =
        fbm(vec3((pos * 10.0), (globals.scene.time * 0.05)), 4, 2.0, 0.5);
    float weight = (1.0 - t);
    weight = (weight * weight);
    vec3 noiseColor =
        vec3((0.5 + (0.5 * sin((((noise * 5.0) + globals.scene.time) + 0.0)))),
             (0.5 + (0.5 * sin((((noise * 5.0) + globals.scene.time) + 2.0)))),
             (0.5 + (0.5 * sin((((noise * 5.0) + globals.scene.time) + 4.0)))));
    color.rgb += (noiseColor * weight);
    totalWeight += weight;
    direction = (mat2(cos((t * 3.0)), (-sin((t * 3.0))), sin((t * 3.0)),
                      cos((t * 3.0))) *
                 direction);
  }
  color.rgb /= totalWeight;
  color.a = 1.0;
  float vignette = (1.0 - smoothstep(0.5, 1.0, (length((uv - 0.5)) * 1.5)));
  color.rgb *= vignette;
  imageStore(outputImage, texCoord, color);
}

#endif
