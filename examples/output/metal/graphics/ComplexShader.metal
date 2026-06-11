
#include <metal_stdlib>
using namespace metal;

__attribute__((unused)) constant float PI = 3.14159265359;
__attribute__((unused)) constant float EPSILON = 0.0001;
__attribute__((unused)) constant int MAX_ITERATIONS = 64;
__attribute__((unused)) constant float3 UP_VECTOR = float3(0.0, 1.0, 0.0);

struct Material {
  float3 albedo;
  float roughness;
  float metallic;
  float3 emissive;
  float opacity;
  bool hasNormalMap;
  texture2d<float> albedoMap;
  texture2d<float> normalMap;
  texture2d<float> metallicRoughnessMap;
};
struct Light {
  float3 position;
  float3 color;
  float intensity;
  float radius;
  bool castShadows;
  float4x4 viewProjection;
};
struct Scene {
  Material materials[4];
  Light lights[8];
  float3 ambientLight;
  float time;
  float elapsedTime;
  int activeLightCount;
  float4x4 viewMatrix;
  float4x4 projectionMatrix;
};
struct VertexInput {
  float3 position [[attribute(0)]];
  float3 normal [[attribute(1)]];
  float3 tangent [[attribute(2)]];
  float3 bitangent [[attribute(3)]];
  float2 texCoord0 [[attribute(4)]];
  float2 texCoord1 [[attribute(5)]];
  float4 color [[attribute(6)]];
  int materialIndex [[attribute(7)]];
};
struct VertexOutput {
  float3 worldPosition;
  float3 worldNormal;
  float3 worldTangent;
  float3 worldBitangent;
  float2 texCoord0;
  float2 texCoord1;
  float4 color;
  float3 TBN_0;
  float3 TBN_1;
  float3 TBN_2;
  int materialIndex;
  float4 clipPosition [[position]];
};
struct FragmentOutput {
  float4 color [[color(0)]];
  float4 normalBuffer [[color(1)]];
  float4 positionBuffer [[color(2)]];
  float depth [[depth(any)]];
};
struct GlobalUniforms {
  Scene scene;
  float3 cameraPosition;
  float globalRoughness;
  float2 screenSize;
  float nearPlane;
  float farPlane;
  int frameCount;
  float noiseValues[1024];
};
static inline float __crossgl_det3_float4x4(float a00, float a01, float a02,
                                            float a10, float a11, float a12,
                                            float a20, float a21, float a22) {
  return a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) +
         a02 * (a10 * a21 - a11 * a20);
}

static inline float __crossgl_cofactor_float4x4(float4x4 m, int column,
                                                int row) {
  float values[9];
  int index = 0;
  for (int c = 0; c < 4; ++c) {
    if (c == column) {
      continue;
    }
    for (int r = 0; r < 4; ++r) {
      if (r == row) {
        continue;
      }
      values[index++] = m[c][r];
    }
  }
  float minor_det = __crossgl_det3_float4x4(values[0], values[1], values[2],
                                            values[3], values[4], values[5],
                                            values[6], values[7], values[8]);
  return ((column + row) & 1) ? -minor_det : minor_det;
}

static inline float4x4 __crossgl_inverse_float4x4(float4x4 m) {
  float det = determinant(m);
  if (abs(det) <= 1.0e-8) {
    return float4x4(1.0);
  }
  float4x4 result;
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      result[c][r] = __crossgl_cofactor_float4x4(m, r, c) / det;
    }
  }
  return result;
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

float noise3D(float3 p) {
  float3 i = floor(p);
  float3 f = fract(p);
  float3 u = f * f * f * f * f * float3(6.0) - float3(15.0) + float3(10.0);
  float n000 = fract(sin(dot(i, float3(13.534, 43.5234, 243.32))) * 4453.0);
  float n001 = fract(
      sin(dot(i + float3(0.0, 0.0, 1.0), float3(13.534, 43.5234, 243.32))) *
      4453.0);
  float n010 = fract(
      sin(dot(i + float3(0.0, 1.0, 0.0), float3(13.534, 43.5234, 243.32))) *
      4453.0);
  float n011 = fract(
      sin(dot(i + float3(0.0, 1.0, 1.0), float3(13.534, 43.5234, 243.32))) *
      4453.0);
  float n100 = fract(
      sin(dot(i + float3(1.0, 0.0, 0.0), float3(13.534, 43.5234, 243.32))) *
      4453.0);
  float n101 = fract(
      sin(dot(i + float3(1.0, 0.0, 1.0), float3(13.534, 43.5234, 243.32))) *
      4453.0);
  float n110 = fract(
      sin(dot(i + float3(1.0, 1.0, 0.0), float3(13.534, 43.5234, 243.32))) *
      4453.0);
  float n111 = fract(
      sin(dot(i + float3(1.0, 1.0, 1.0), float3(13.534, 43.5234, 243.32))) *
      4453.0);
  float n00 = mix(n000, n001, u.z);
  float n01 = mix(n010, n011, u.z);
  float n10 = mix(n100, n101, u.z);
  float n11 = mix(n110, n111, u.z);
  float n0 = mix(n00, n01, u.y);
  float n1 = mix(n10, n11, u.y);
  return mix(n0, n1, u.x);
}

float fbm(float3 p, int octaves, float lacunarity, float gain) {
  float sum = 0.0;
  float amplitude = 1.0;
  float frequency = 1.0;
  for (int i = 0; i < octaves; ++i) {
    if (i >= MAX_ITERATIONS) {
      break;
    }
    sum += amplitude * noise3D(p * frequency);
    amplitude *= gain;
    frequency *= lacunarity;
  }
  return sum;
}

float4 samplePlanarProjection(texture2d<float> tex, float3 worldPos,
                              float3 normal) {
  float3 absNormal = abs(normal);
  bool useX = absNormal.x >= absNormal.y && absNormal.x >= absNormal.z;
  bool useY = !useX && absNormal.y >= absNormal.z;
  float2 uv;
  if (useX) {
    uv = worldPos.zy * float2(0.5) + float2(0.5);
    if (normal.x < 0.0) {
      uv.x = 1.0 - uv.x;
    }
  } else if (useY) {
    uv = worldPos.xz * float2(0.5) + float2(0.5);
    if (normal.y < 0.0) {
      uv.y = 1.0 - uv.y;
    }
  } else {
    uv = worldPos.xy * float2(0.5) + float2(0.5);
    if (normal.z < 0.0) {
      uv.x = 1.0 - uv.x;
    }
  }
  return tex.sample(sampler(mag_filter::linear, min_filter::linear), uv);
}

// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]],
                                constant GlobalUniforms &globals [[buffer(0)]],
                                texture2d<float> shadowMap [[texture(0)]]) {
  VertexOutput output;
  float4x4 modelMatrix = float4x4(1.0);
  float4x4 viewMatrix = globals.scene.viewMatrix;
  float4x4 projectionMatrix = globals.scene.projectionMatrix;
  float4x4 modelViewMatrix = viewMatrix * modelMatrix;
  float4x4 modelViewProjectionMatrix = projectionMatrix * modelViewMatrix;
  float3x3 normalMatrix =
      float3x3(transpose(__crossgl_inverse_float4x4(modelMatrix))[0].xyz,
               transpose(__crossgl_inverse_float4x4(modelMatrix))[1].xyz,
               transpose(__crossgl_inverse_float4x4(modelMatrix))[2].xyz);
  float4 worldPosition = modelMatrix * float4(input.position, 1.0);
  float3 worldNormal = normalize(normalMatrix * input.normal);
  float3 worldTangent = normalize(normalMatrix * input.tangent);
  float3 worldBitangent = normalize(normalMatrix * input.bitangent);
  float3x3 TBN = float3x3(worldTangent, worldBitangent, worldNormal);
  float displacement =
      fbm(worldPosition.xyz + globals.scene.time * 0.1, 4, 2.0, 0.5) * 0.1;
  if (input.materialIndex > 0) {
    worldPosition.xyz += worldNormal * float3(displacement);
  }
  float3 viewDir = normalize(globals.cameraPosition - worldPosition.xyz);
  float fresnel = pow(1.0 - max(0.0, dot(worldNormal, viewDir)), 5.0);
  if (input.materialIndex < globals.scene.activeLightCount) {
    output.color = input.color * float4(1.0, 1.0, 1.0, 1.0);
    for (int i = 0; i < 4; ++i) {
      if (i >= globals.frameCount % 5) {
        break;
      }
      Light light = globals.scene.lights[i];
      float3 lightDir = normalize(light.position - worldPosition.xyz);
      float lightDistance = length(light.position - worldPosition.xyz);
      float attenuation = 1.0 / 1.0 + lightDistance * lightDistance;
      float lightIntensity = light.intensity * attenuation;
      output.color.rgb += light.color * float3(lightIntensity) *
                          float3(max(0.0, dot(worldNormal, lightDir))) *
                          float3(0.025);
    }
  } else {
    output.color = input.color;
    if (globals.globalRoughness > 0.5) {
      if (fresnel > 0.7) {
        output.color.a *= 0.8;
      } else {
        output.color.a *= 0.9;
      }
    }
  }
  output.worldPosition = worldPosition.xyz;
  output.worldNormal = worldNormal;
  output.worldTangent = worldTangent;
  output.worldBitangent = worldBitangent;
  output.texCoord0 = input.texCoord0;
  output.texCoord1 = input.texCoord1;
  float3x3 __crossgl_stage_io_matrix_0 = TBN;
  output.TBN_0 = __crossgl_stage_io_matrix_0[0];
  output.TBN_1 = __crossgl_stage_io_matrix_0[1];
  output.TBN_2 = __crossgl_stage_io_matrix_0[2];
  output.materialIndex = input.materialIndex;
  output.clipPosition = modelViewProjectionMatrix * float4(input.position, 1.0);
  return output;
}

float shadowCalculation(float4 fragPosLightSpace, int iteration,
                        VertexOutput input, texture2d<float> shadowMap,
                        constant GlobalUniforms &globals);

float shadowCalculation(float4 fragPosLightSpace, int iteration,
                        VertexOutput input, texture2d<float> shadowMap,
                        constant GlobalUniforms &globals) {
  if (iteration > 3) {
    return 0.0;
  }
  float3 projCoords = fragPosLightSpace.xyz / float3(fragPosLightSpace.w);
  projCoords = projCoords * float3(0.5) + float3(0.5);
  float closestDepth =
      shadowMap
          .sample(sampler(mag_filter::linear, min_filter::linear),
                  projCoords.xy)
          .r;
  float currentDepth = projCoords.z;
  float bias =
      max(0.05 * 1.0 - dot(input.worldNormal, normalize(globals.cameraPosition -
                                                        input.worldPosition)),
          0.005);
  float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
  __attribute__((unused)) float pcfDepth = 0.0;
  float2 texelSize = float2(1.0) / float2(globals.screenSize);
  float offset = globals.noiseValues[iteration * 4 % 16] * 0.001;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      float pcfDepth =
          shadowMap
              .sample(sampler(mag_filter::linear, min_filter::linear),
                      projCoords.xy + float2(x, y) * texelSize + float2(offset))
              .r;
      shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
    }
  }
  shadow /= 9.0;
  if (projCoords.z > 1.0) {
    shadow = 0.0;
  }
  return shadow;
}

// Fragment Shader
fragment FragmentOutput
fragment_main(VertexOutput input [[stage_in]],
              constant GlobalUniforms &globals [[buffer(0)]],
              texture2d<float> shadowMap [[texture(0)]],
              texture2d<float, access::read_write> outputImage [[texture(1)]]) {
  FragmentOutput output;
  Material material = globals.scene.materials[input.materialIndex];
  float4 albedoValue = material.albedoMap.sample(
      sampler(mag_filter::linear, min_filter::linear), input.texCoord0);
  float4 normalValue = material.normalMap.sample(
      sampler(mag_filter::linear, min_filter::linear), input.texCoord0);
  float4 metallicRoughnessValue = material.metallicRoughnessMap.sample(
      sampler(mag_filter::linear, min_filter::linear), input.texCoord0);
  float3 normal = normalValue.xyz * float3(2.0) - float3(1.0);
  float3 worldNormal =
      normalize(float3x3(input.TBN_0, input.TBN_1, input.TBN_2) * normal);
  float3 albedo = albedoValue.rgb * material.albedo;
  float metallic = metallicRoughnessValue.b * material.metallic;
  float roughness = metallicRoughnessValue.g * material.roughness;
  float ao = metallicRoughnessValue.r;
  float3 viewDir = normalize(globals.cameraPosition - input.worldPosition);
  float3 F0 = mix(float3(0.04), albedo, metallic);
  float3 Lo = float3(0.0);
  for (int i = 0; i < globals.scene.activeLightCount; ++i) {
    if (i >= 8) {
      break;
    }
    Light light = globals.scene.lights[i];
    float3 lightDir = normalize(light.position - input.worldPosition);
    float3 halfway = normalize(viewDir + lightDir);
    float distance = length(light.position - input.worldPosition);
    float attenuation = 1.0 / distance * distance;
    float3 radiance =
        light.color * float3(light.intensity) * float3(attenuation);
    float NDF = distributionGGX(worldNormal, halfway, roughness);
    float G = geometrySmith(worldNormal, viewDir, lightDir, roughness);
    float3 F = fresnelSchlick(max(dot(halfway, viewDir), 0.0), F0);
    float3 kS = F;
    float3 kD = float3(1.0) - kS;
    kD *= 1.0 - metallic;
    float3 numerator = float3(NDF * G) * F;
    float denominator = 4.0 * max(dot(worldNormal, viewDir), 0.0) *
                            max(dot(worldNormal, lightDir), 0.0) +
                        EPSILON;
    float3 specular = numerator / float3(denominator);
    float NdotL = max(dot(worldNormal, lightDir), 0.0);
    float shadow = 0.0;
    if (light.castShadows) {
      float4 fragPosLightSpace =
          light.viewProjection * float4(input.worldPosition, 1.0);
      shadow =
          shadowCalculation(fragPosLightSpace, 0, input, shadowMap, globals);
      for (int s = 0; s < 4; ++s) {
        if (s >= globals.frameCount % 3) {
          continue;
        }
        shadow += shadowCalculation(
            fragPosLightSpace +
                float4(globals.noiseValues[s % 16] * 0.001, 0.0, 0.0, 0.0),
            s + 1, input, shadowMap, globals);
      }
      shadow /= 5.0;
    }
    Lo += float3(1.0 - shadow) * kD * albedo / PI +
          specular * radiance * float3(NdotL);
  }
  float3 ambient = globals.scene.ambientLight * albedo * float3(ao);
  float3 color = ambient + Lo;
  color = color / color + float3(1.0);
  color = pow(color, float3(1.0 / 2.2));
  output.color = float4(color, material.opacity * albedoValue.a);
  output.normalBuffer = float4(worldNormal * 0.5 + 0.5, 1.0);
  output.positionBuffer = float4(input.worldPosition, 1.0);
  output.depth = input.clipPosition.z / input.clipPosition.w;
  return output;
}

// Compute Shader
kernel void kernel_main(uint3 thread_position_in_grid
                        [[thread_position_in_grid]],
                        constant GlobalUniforms &globals [[buffer(0)]],
                        texture2d<float> shadowMap [[texture(0)]],
                        texture2d<float, access::read_write> outputImage
                        [[texture(1)]]) {
  int2 texCoord = int2(thread_position_in_grid.xy);
  float2 screenSize = globals.screenSize;
  if (texCoord.x >= int(screenSize.x) || texCoord.y >= int(screenSize.y)) {
    return;
  }
  float2 uv = float2(texCoord) / screenSize;
  float4 color = float4(0.0);
  float totalWeight = 0.0;
  float2 direction = float2(0.5) - uv;
  float len = length(direction);
  direction = normalize(direction);
  for (int i = 0; i < 32; ++i) {
    if (i >= MAX_ITERATIONS) {
      break;
    }
    float t = float(i) / 32.0;
    float2 pos = uv + direction * float2(t) * float2(len) * float2(0.1);
    float noise =
        fbm(float3(pos * 10.0, globals.scene.time * 0.05), 4, 2.0, 0.5);
    float weight = 1.0 - t;
    weight = weight * weight;
    float3 noiseColor =
        float3(0.5 + 0.5 * sin(noise * 5.0 + globals.scene.time + 0.0),
               0.5 + 0.5 * sin(noise * 5.0 + globals.scene.time + 2.0),
               0.5 + 0.5 * sin(noise * 5.0 + globals.scene.time + 4.0));
    color.rgb += noiseColor * float3(weight);
    totalWeight += weight;
    direction =
        float2x2(cos(t * 3.0), -sin(t * 3.0), sin(t * 3.0), cos(t * 3.0)) *
        direction;
  }
  color.rgb /= totalWeight;
  color.a = 1.0;
  float vignette = 1.0 - smoothstep(0.5, 1.0, length(uv - 0.5) * 1.5);
  color.rgb *= vignette;
  outputImage.write(color, uint2(texCoord));
}
