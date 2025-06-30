#include <hip/device_functions.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/math_functions.h>

struct Material {
  float3 albedo;
  float roughness;
  float metallic;
  float3 emissive;
  float opacity;
  bool hasNormalMap;
  texture<float4, 2> albedoMap;
  texture<float4, 2> normalMap;
  texture<float4, 2> metallicRoughnessMap;
};

struct Light {
  float3 position;
  float3 color;
  float intensity;
  float radius;
  bool castShadows;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) viewProjection;
};

struct Scene {
  Material[LiteralNode(value = 4, literal_type = PrimitiveType(
                                      name = int, size_bits = None))] materials;
  Light[LiteralNode(value = 8, literal_type = PrimitiveType(
                                   name = int, size_bits = None))] lights;
  float3 ambientLight;
  float time;
  float elapsedTime;
  int activeLightCount;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) viewMatrix;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) projectionMatrix;
};

struct VertexInput {
  float3 position;
  float3 normal;
  float3 tangent;
  float3 bitangent;
  float2 texCoord0;
  float2 texCoord1;
  float4 color;
  int materialIndex;
};

struct VertexOutput {
  float3 worldPosition;
  float3 worldNormal;
  float3 worldTangent;
  float3 worldBitangent;
  float2 texCoord0;
  float2 texCoord1;
  float4 color;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 3, cols = 3) TBN;
  int materialIndex;
  float4 clipPosition;
};

struct FragmentOutput {
  float4 color;
  float4 normalBuffer;
  float4 positionBuffer;
  float depth;
};

struct GlobalUniforms {
  Scene scene;
  float3 cameraPosition;
  float globalRoughness;
  float2 screenSize;
  float nearPlane;
  float farPlane;
  int frameCount;
  float[] noiseValues;
};

__device__ float distributionGGX(
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

__device__ float geometrySchlickGGX(float NdotV, float roughness) {
  float r = (roughness + 1.0);
  float k = ((r * r) / 8.0);
  float num = NdotV;
  float denom = ((NdotV * (1.0 - k)) + k);
  return (num / IdentifierNode(name = max)(denom, EPSILON));
}

__device__ float geometrySmith(
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

__device__ VectorType(element_type = PrimitiveType(name = float,
                                                   size_bits = None),
                      size = 3)
    fresnelSchlick(float cosTheta,
                   VectorType(element_type = PrimitiveType(name = float,
                                                           size_bits = None),
                              size = 3) F0) {
  return (F0 + ((1.0 - F0) *
                IdentifierNode(name = pow)(
                    IdentifierNode(name = max)((1.0 - cosTheta), 0.0), 5.0)));
}

__device__ float noise3D(VectorType(
    element_type = PrimitiveType(name = float, size_bits = None), size = 3) p) {
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) i = IdentifierNode(name = floor)(p);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) f = IdentifierNode(name = fract)(p);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) u = (((f * f) * f) * ((f * ((f * 6.0) - 15.0)) + 10.0));
  float n000 = IdentifierNode(name = fract)(
      (IdentifierNode(name = sin)(IdentifierNode(name = dot)(
           i, IdentifierNode(name = vec3)(13.534, 43.5234, 243.32))) *
       4453.0));
  float n001 = IdentifierNode(name = fract)(
      (IdentifierNode(name = sin)(IdentifierNode(name = dot)(
           (i + IdentifierNode(name = vec3)(0.0, 0.0, 1.0)),
           IdentifierNode(name = vec3)(13.534, 43.5234, 243.32))) *
       4453.0));
  float n010 = IdentifierNode(name = fract)(
      (IdentifierNode(name = sin)(IdentifierNode(name = dot)(
           (i + IdentifierNode(name = vec3)(0.0, 1.0, 0.0)),
           IdentifierNode(name = vec3)(13.534, 43.5234, 243.32))) *
       4453.0));
  float n011 = IdentifierNode(name = fract)(
      (IdentifierNode(name = sin)(IdentifierNode(name = dot)(
           (i + IdentifierNode(name = vec3)(0.0, 1.0, 1.0)),
           IdentifierNode(name = vec3)(13.534, 43.5234, 243.32))) *
       4453.0));
  float n100 = IdentifierNode(name = fract)(
      (IdentifierNode(name = sin)(IdentifierNode(name = dot)(
           (i + IdentifierNode(name = vec3)(1.0, 0.0, 0.0)),
           IdentifierNode(name = vec3)(13.534, 43.5234, 243.32))) *
       4453.0));
  float n101 = IdentifierNode(name = fract)(
      (IdentifierNode(name = sin)(IdentifierNode(name = dot)(
           (i + IdentifierNode(name = vec3)(1.0, 0.0, 1.0)),
           IdentifierNode(name = vec3)(13.534, 43.5234, 243.32))) *
       4453.0));
  float n110 = IdentifierNode(name = fract)(
      (IdentifierNode(name = sin)(IdentifierNode(name = dot)(
           (i + IdentifierNode(name = vec3)(1.0, 1.0, 0.0)),
           IdentifierNode(name = vec3)(13.534, 43.5234, 243.32))) *
       4453.0));
  float n111 = IdentifierNode(name = fract)(
      (IdentifierNode(name = sin)(IdentifierNode(name = dot)(
           (i + IdentifierNode(name = vec3)(1.0, 1.0, 1.0)),
           IdentifierNode(name = vec3)(13.534, 43.5234, 243.32))) *
       4453.0));
  float n00 = IdentifierNode(name = mix)(n000, n001, u.z);
  float n01 = IdentifierNode(name = mix)(n010, n011, u.z);
  float n10 = IdentifierNode(name = mix)(n100, n101, u.z);
  float n11 = IdentifierNode(name = mix)(n110, n111, u.z);
  float n0 = IdentifierNode(name = mix)(n00, n01, u.y);
  float n1 = IdentifierNode(name = mix)(n10, n11, u.y);
  return IdentifierNode(name = mix)(n0, n1, u.x);
}

__device__ float fbm(VectorType(element_type = PrimitiveType(name = float,
                                                             size_bits = None),
                                size = 3) p,
                     int octaves, float lacunarity, float gain) {
  float sum = 0.0;
  float amplitude = 1.0;
  float frequency = 1.0;
  int i = 0;
  for (; (i < octaves); ++i) {
    if ((i >= MAX_ITERATIONS)) {
      break;
    }
    sum = (amplitude * IdentifierNode(name = noise3D)((p * frequency)));
    amplitude = gain;
    frequency = lacunarity;
  }
  return sum;
}

__device__ VectorType(element_type = PrimitiveType(name = float,
                                                   size_bits = None),
                      size = 4)
    samplePlanarProjection(
        texture<float4, 2> tex,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) worldPos,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) normal) {
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) absNormal = IdentifierNode(name = abs)(normal);
  bool useX = ((absNormal.x >= absNormal.y) && (absNormal.x >= absNormal.z));
  bool useY = (!useX && (absNormal.y >= absNormal.z));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 2) uv;
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
  return IdentifierNode(name = texture)(tex, uv);
}

__device__ VertexOutput main(VertexInput input) {
  VertexOutput output;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) modelMatrix = IdentifierNode(name = mat4)(1.0);
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) viewMatrix = globals.scene.viewMatrix;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) projectionMatrix =
      globals.scene.projectionMatrix;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) modelViewMatrix = (viewMatrix * modelMatrix);
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) modelViewProjectionMatrix =
      (projectionMatrix * modelViewMatrix);
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 3, cols = 3) normalMatrix =
      IdentifierNode(name = mat3)(IdentifierNode(name = transpose)(
          IdentifierNode(name = inverse)(modelMatrix)));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) worldPosition =
      (modelMatrix * IdentifierNode(name = vec4)(input.position, 1.0));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) worldNormal =
      IdentifierNode(name = normalize)((normalMatrix * input.normal));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) worldTangent =
      IdentifierNode(name = normalize)((normalMatrix * input.tangent));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) worldBitangent =
      IdentifierNode(name = normalize)((normalMatrix * input.bitangent));
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 3, cols = 3) TBN =
      IdentifierNode(name = mat3)(worldTangent, worldBitangent, worldNormal);
  float displacement =
      (IdentifierNode(name = fbm)(
           (worldPosition.xyz + (globals.scene.time * 0.1)), 4, 2.0, 0.5) *
       0.1);
  if ((input.materialIndex > 0)) {
    worldPosition.xyz = (worldNormal * displacement);
  }
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) viewDir =
      IdentifierNode(name = normalize)(
          (globals.cameraPosition - worldPosition.xyz));
  float fresnel = IdentifierNode(name = pow)(
      (1.0 - IdentifierNode(name = max)(
                 0.0, IdentifierNode(name = dot)(worldNormal, viewDir))),
      5.0);
  if ((input.materialIndex < globals.scene.activeLightCount)) {
    output.color =
        (input.color * IdentifierNode(name = vec4)(1.0, 1.0, 1.0, 1.0));
    int i = 0;
    for (; (i < 4); ++i) {
      if ((i >= (globals.frameCount % 5))) {
        break;
      }
      Light light = globals.scene.lights[i];
      VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                 size = 3) lightDir =
          IdentifierNode(name =
                             normalize)((light.position - worldPosition.xyz));
      float lightDistance =
          IdentifierNode(name = length)((light.position - worldPosition.xyz));
      float attenuation = (1.0 / (1.0 + (lightDistance * lightDistance)));
      float lightIntensity = (light.intensity * attenuation);
      output.color.rgb =
          (((light.color * lightIntensity) *
            IdentifierNode(name = max)(
                0.0, IdentifierNode(name = dot)(worldNormal, lightDir))) *
           0.025);
    }
  } else {
    output.color = input.color;
    if ((globals.globalRoughness > 0.5)) {
      if ((fresnel > 0.7)) {
        output.color.a = 0.8;
      } else {
        output.color.a = 0.9;
      }
    }
  }
  output.worldPosition = worldPosition.xyz;
  output.worldNormal = worldNormal;
  output.worldTangent = worldTangent;
  output.worldBitangent = worldBitangent;
  output.texCoord0 = input.texCoord0;
  output.texCoord1 = input.texCoord1;
  output.TBN = TBN;
  output.materialIndex = input.materialIndex;
  output.clipPosition = (modelViewProjectionMatrix *
                         IdentifierNode(name = vec4)(input.position, 1.0));
  return output;
}

__device__ FragmentOutput main(VertexOutput input) {
  FragmentOutput output;
  Material material = globals.scene.materials[input.materialIndex];
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) albedoValue =
      IdentifierNode(name = texture)(material.albedoMap, input.texCoord0);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) normalValue =
      IdentifierNode(name = texture)(material.normalMap, input.texCoord0);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) metallicRoughnessValue =
      IdentifierNode(name = texture)(material.metallicRoughnessMap,
                                     input.texCoord0);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) normal = ((normalValue.xyz * 2.0) - 1.0);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) worldNormal =
      IdentifierNode(name = normalize)((input.TBN * normal));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) albedo = (albedoValue.rgb * material.albedo);
  float metallic = (metallicRoughnessValue.b * material.metallic);
  float roughness = (metallicRoughnessValue.g * material.roughness);
  float ao = metallicRoughnessValue.r;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) viewDir =
      IdentifierNode(name = normalize)(
          (globals.cameraPosition - input.worldPosition));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) F0 =
      IdentifierNode(name = mix)(IdentifierNode(name = vec3)(0.04), albedo,
                                 metallic);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) Lo = IdentifierNode(name = vec3)(0.0);
  int i = 0;
  for (; (i < globals.scene.activeLightCount); ++i) {
    if ((i >= 8)) {
      break;
    }
    Light light = globals.scene.lights[i];
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) lightDir =
        IdentifierNode(name =
                           normalize)((light.position - input.worldPosition));
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) halfway =
        IdentifierNode(name = normalize)((viewDir + lightDir));
    float distance =
        IdentifierNode(name = length)((light.position - input.worldPosition));
    float attenuation = (1.0 / (distance * distance));
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) radiance =
        ((light.color * light.intensity) * attenuation);
    float NDF =
        IdentifierNode(name = distributionGGX)(worldNormal, halfway, roughness);
    float G = IdentifierNode(name = geometrySmith)(worldNormal, viewDir,
                                                   lightDir, roughness);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) F =
        IdentifierNode(name = fresnelSchlick)(
            IdentifierNode(name = max)(
                IdentifierNode(name = dot)(halfway, viewDir), 0.0),
            F0);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) kS = F;
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) kD = (IdentifierNode(name = vec3)(1.0) - kS);
    kD = (1.0 - metallic);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) numerator = ((NDF * G) * F);
    float denominator =
        (((4.0 * IdentifierNode(name = max)(
                     IdentifierNode(name = dot)(worldNormal, viewDir), 0.0)) *
          IdentifierNode(name = max)(
              IdentifierNode(name = dot)(worldNormal, lightDir), 0.0)) +
         EPSILON);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) specular = (numerator / denominator);
    float NdotL = IdentifierNode(name = max)(
        IdentifierNode(name = dot)(worldNormal, lightDir), 0.0);
    float shadow = 0.0;
    if (light.castShadows) {
      VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                 size = 4) fragPosLightSpace =
          (light.viewProjection *
           IdentifierNode(name = vec4)(input.worldPosition, 1.0));
      shadow = IdentifierNode(name = shadowCalculation)(fragPosLightSpace, 0);
      int s = 0;
      for (; (s < 4); ++s) {
        if ((s >= (globals.frameCount % 3))) {
          continue;
        }
        shadow = IdentifierNode(name = shadowCalculation)(
            (fragPosLightSpace +
             IdentifierNode(name = vec4)(
                 (globals.noiseValues[(s % 16)] * 0.001), 0.0, 0.0, 0.0)),
            (s + 1));
      }
      shadow = 5.0;
    }
    Lo = ((((1.0 - shadow) * (((kD * albedo) / PI) + specular)) * radiance) *
          NdotL);
  }
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) ambient = ((globals.scene.ambientLight * albedo) * ao);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) color = (ambient + Lo);
  color = (color / (color + IdentifierNode(name = vec3)(1.0)));
  color = IdentifierNode(name = pow)(color,
                                     IdentifierNode(name = vec3)((1.0 / 2.2)));
  output.color =
      IdentifierNode(name = vec4)(color, (material.opacity * albedoValue.a));
  output.normalBuffer =
      IdentifierNode(name = vec4)(((worldNormal * 0.5) + 0.5), 1.0);
  output.positionBuffer = IdentifierNode(name = vec4)(input.worldPosition, 1.0);
  output.depth = (input.clipPosition.z / input.clipPosition.w);
  return output;
}

__device__ float shadowCalculation(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 4) fragPosLightSpace,
    int iteration);

__device__ float shadowCalculation(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 4) fragPosLightSpace,
    int iteration) {
  if ((iteration > 3)) {
    return 0.0;
  }
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) projCoords =
      (fragPosLightSpace.xyz / fragPosLightSpace.w);
  projCoords = ((projCoords * 0.5) + 0.5);
  float closestDepth =
      IdentifierNode(name = texture)(shadowMap, projCoords.xy).r;
  float currentDepth = projCoords.z;
  float bias = IdentifierNode(name = max)(
      (0.05 * (1.0 - IdentifierNode(name = dot)(
                         input.worldNormal,
                         IdentifierNode(name = normalize)(
                             (globals.cameraPosition - input.worldPosition))))),
      0.005);
  float shadow = (((currentDepth - bias) > closestDepth) ? 1.0 : 0.0);
  float pcfDepth = 0.0;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 2) texelSize =
      (1.0 / IdentifierNode(name = vec2)(globals.screenSize));
  float offset = (globals.noiseValues[((iteration * 4) % 16)] * 0.001);
  int x = -1;
  for (; (x <= 1); ++x) {
    int y = -1;
    for (; (y <= 1); ++y) {
      float pcfDepth =
          IdentifierNode(name = texture)(
              shadowMap, ((projCoords.xy +
                           (IdentifierNode(name = vec2)(x, y) * texelSize)) +
                          IdentifierNode(name = vec2)(offset)))
              .r;
      shadow = (((currentDepth - bias) > pcfDepth) ? 1.0 : 0.0);
    }
  }
  shadow = 9.0;
  if ((projCoords.z > 1.0)) {
    shadow = 0.0;
  }
  return shadow;
}

__global__ void main() {
  VectorType(element_type = PrimitiveType(name = int, size_bits = None),
             size = 2) texCoord =
      IdentifierNode(name = ivec2)(gl_GlobalInvocationID.xy);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 2) screenSize = globals.screenSize;
  if (((texCoord.x >= IdentifierNode(name = int)(screenSize.x)) ||
       (texCoord.y >= IdentifierNode(name = int)(screenSize.y)))) {
    return;
  }
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 2) uv =
      (IdentifierNode(name = vec2)(texCoord) / screenSize);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) color = IdentifierNode(name = vec4)(0.0);
  float totalWeight = 0.0;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 2) direction = (IdentifierNode(name = vec2)(0.5) - uv);
  float len = IdentifierNode(name = length)(direction);
  direction = IdentifierNode(name = normalize)(direction);
  int i = 0;
  for (; (i < 32); ++i) {
    if ((i >= MAX_ITERATIONS)) {
      break;
    }
    float t = (IdentifierNode(name = float)(i) / 32.0);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 2) pos = (uv + (((direction * t) * len) * 0.1));
    float noise = IdentifierNode(name = fbm)(
        IdentifierNode(name = vec3)((pos * 10.0), (globals.scene.time * 0.05)),
        4, 2.0, 0.5);
    float weight = (1.0 - t);
    weight = (weight * weight);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) noiseColor =
        IdentifierNode(name = vec3)(
            (0.5 + (0.5 * IdentifierNode(name = sin)(
                              (((noise * 5.0) + globals.scene.time) + 0.0)))),
            (0.5 + (0.5 * IdentifierNode(name = sin)(
                              (((noise * 5.0) + globals.scene.time) + 2.0)))),
            (0.5 + (0.5 * IdentifierNode(name = sin)(
                              (((noise * 5.0) + globals.scene.time) + 4.0)))));
    color.rgb = (noiseColor * weight);
    totalWeight = weight;
    direction =
        (IdentifierNode(name = mat2)(IdentifierNode(name = cos)((t * 3.0)),
                                     -IdentifierNode(name = sin)((t * 3.0)),
                                     IdentifierNode(name = sin)((t * 3.0)),
                                     IdentifierNode(name = cos)((t * 3.0))) *
         direction);
  }
  color.rgb = totalWeight;
  color.a = 1.0;
  float vignette =
      (1.0 - IdentifierNode(name = smoothstep)(
                 0.5, 1.0, (IdentifierNode(name = length)((uv - 0.5)) * 1.5)));
  color.rgb = vignette;
  IdentifierNode(name = imageStore)(outputImage, texCoord, color);
}
