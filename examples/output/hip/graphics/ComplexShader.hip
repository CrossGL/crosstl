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
  float NdotH = fmaxf(dot(N, H), 0.0);
  float NdotH2 = (NdotH * NdotH);
  float num = a2;
  float denom = ((NdotH2 * (a2 - 1.0)) + 1.0);
  denom = ((PI * denom) * denom);
  return (num / fmaxf(denom, EPSILON));
}

__device__ float geometrySchlickGGX(float NdotV, float roughness) {
  float r = (roughness + 1.0);
  float k = ((r * r) / 8.0);
  float num = NdotV;
  float denom = ((NdotV * (1.0 - k)) + k);
  return (num / fmaxf(denom, EPSILON));
}

__device__ float geometrySmith(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) N,
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) V,
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) L,
    float roughness) {
  float NdotV = fmaxf(dot(N, V), 0.0);
  float NdotL = fmaxf(dot(N, L), 0.0);
  float ggx2 = geometrySchlickGGX(NdotV, roughness);
  float ggx1 = geometrySchlickGGX(NdotL, roughness);
  return (ggx1 * ggx2);
}

__device__ VectorType(element_type = PrimitiveType(name = float,
                                                   size_bits = None),
                      size = 3)
    fresnelSchlick(float cosTheta,
                   VectorType(element_type = PrimitiveType(name = float,
                                                           size_bits = None),
                              size = 3) F0) {
  return (F0 + ((1.0 - F0) * powf(fmaxf((1.0 - cosTheta), 0.0), 5.0)));
}

__device__ float noise3D(VectorType(
    element_type = PrimitiveType(name = float, size_bits = None), size = 3) p) {
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) i = floorf(p);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) f = fracf(p);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) u = (((f * f) * f) * ((f * ((f * 6.0) - 15.0)) + 10.0));
  float n000 =
      fracf((sinf(dot(i, make_float3(13.534, 43.5234, 243.32))) * 4453.0));
  float n001 = fracf((sinf(dot((i + make_float3(0.0, 0.0, 1.0)),
                               make_float3(13.534, 43.5234, 243.32))) *
                      4453.0));
  float n010 = fracf((sinf(dot((i + make_float3(0.0, 1.0, 0.0)),
                               make_float3(13.534, 43.5234, 243.32))) *
                      4453.0));
  float n011 = fracf((sinf(dot((i + make_float3(0.0, 1.0, 1.0)),
                               make_float3(13.534, 43.5234, 243.32))) *
                      4453.0));
  float n100 = fracf((sinf(dot((i + make_float3(1.0, 0.0, 0.0)),
                               make_float3(13.534, 43.5234, 243.32))) *
                      4453.0));
  float n101 = fracf((sinf(dot((i + make_float3(1.0, 0.0, 1.0)),
                               make_float3(13.534, 43.5234, 243.32))) *
                      4453.0));
  float n110 = fracf((sinf(dot((i + make_float3(1.0, 1.0, 0.0)),
                               make_float3(13.534, 43.5234, 243.32))) *
                      4453.0));
  float n111 = fracf((sinf(dot((i + make_float3(1.0, 1.0, 1.0)),
                               make_float3(13.534, 43.5234, 243.32))) *
                      4453.0));
  float n00 = lerp(n000, n001, u.z);
  float n01 = lerp(n010, n011, u.z);
  float n10 = lerp(n100, n101, u.z);
  float n11 = lerp(n110, n111, u.z);
  float n0 = lerp(n00, n01, u.y);
  float n1 = lerp(n10, n11, u.y);
  return lerp(n0, n1, u.x);
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
    sum = (amplitude * noise3D((p * frequency)));
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
             size = 3) absNormal = fabsf(normal);
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
  return tex2D(tex, uv);
}

__device__ VertexOutput main(VertexInput input) {
  VertexOutput output;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) modelMatrix = mat4(1.0);
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
      mat3(transpose(inverse(modelMatrix)));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) worldPosition =
      (modelMatrix * make_float4(input.position, 1.0));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) worldNormal = normalize((normalMatrix * input.normal));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) worldTangent = normalize((normalMatrix * input.tangent));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) worldBitangent =
      normalize((normalMatrix * input.bitangent));
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 3, cols = 3) TBN =
      mat3(worldTangent, worldBitangent, worldNormal);
  float displacement =
      (fbm((worldPosition.xyz + (globals.scene.time * 0.1)), 4, 2.0, 0.5) *
       0.1);
  if ((input.materialIndex > 0)) {
    worldPosition.xyz = (worldNormal * displacement);
  }
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) viewDir =
      normalize((globals.cameraPosition - worldPosition.xyz));
  float fresnel = powf((1.0 - fmaxf(0.0, dot(worldNormal, viewDir))), 5.0);
  if ((input.materialIndex < globals.scene.activeLightCount)) {
    output.color = (input.color * make_float4(1.0, 1.0, 1.0, 1.0));
    int i = 0;
    for (; (i < 4); ++i) {
      if ((i >= (globals.frameCount % 5))) {
        break;
      }
      Light light = globals.scene.lights[i];
      VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                 size = 3) lightDir =
          normalize((light.position - worldPosition.xyz));
      float lightDistance = length((light.position - worldPosition.xyz));
      float attenuation = (1.0 / (1.0 + (lightDistance * lightDistance)));
      float lightIntensity = (light.intensity * attenuation);
      output.color.rgb = (((light.color * lightIntensity) *
                           fmaxf(0.0, dot(worldNormal, lightDir))) *
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
  output.clipPosition =
      (modelViewProjectionMatrix * make_float4(input.position, 1.0));
  return output;
}

__device__ FragmentOutput main(VertexOutput input) {
  FragmentOutput output;
  Material material = globals.scene.materials[input.materialIndex];
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) albedoValue = tex2D(material.albedoMap, input.texCoord0);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) normalValue = tex2D(material.normalMap, input.texCoord0);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) metallicRoughnessValue =
      tex2D(material.metallicRoughnessMap, input.texCoord0);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) normal = ((normalValue.xyz * 2.0) - 1.0);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) worldNormal = normalize((input.TBN * normal));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) albedo = (albedoValue.rgb * material.albedo);
  float metallic = (metallicRoughnessValue.b * material.metallic);
  float roughness = (metallicRoughnessValue.g * material.roughness);
  float ao = metallicRoughnessValue.r;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) viewDir =
      normalize((globals.cameraPosition - input.worldPosition));
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) F0 = lerp(make_float3(0.04), albedo, metallic);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) Lo = make_float3(0.0);
  int i = 0;
  for (; (i < globals.scene.activeLightCount); ++i) {
    if ((i >= 8)) {
      break;
    }
    Light light = globals.scene.lights[i];
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) lightDir =
        normalize((light.position - input.worldPosition));
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) halfway = normalize((viewDir + lightDir));
    float distance = length((light.position - input.worldPosition));
    float attenuation = (1.0 / (distance * distance));
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) radiance =
        ((light.color * light.intensity) * attenuation);
    float NDF = distributionGGX(worldNormal, halfway, roughness);
    float G = geometrySmith(worldNormal, viewDir, lightDir, roughness);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) F =
        fresnelSchlick(fmaxf(dot(halfway, viewDir), 0.0), F0);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) kS = F;
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) kD = (make_float3(1.0) - kS);
    kD = (1.0 - metallic);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) numerator = ((NDF * G) * F);
    float denominator = (((4.0 * fmaxf(dot(worldNormal, viewDir), 0.0)) *
                          fmaxf(dot(worldNormal, lightDir), 0.0)) +
                         EPSILON);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) specular = (numerator / denominator);
    float NdotL = fmaxf(dot(worldNormal, lightDir), 0.0);
    float shadow = 0.0;
    if (light.castShadows) {
      VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                 size = 4) fragPosLightSpace =
          (light.viewProjection * make_float4(input.worldPosition, 1.0));
      shadow = shadowCalculation(fragPosLightSpace, 0);
      int s = 0;
      for (; (s < 4); ++s) {
        if ((s >= (globals.frameCount % 3))) {
          continue;
        }
        shadow = shadowCalculation(
            (fragPosLightSpace +
             make_float4((globals.noiseValues[(s % 16)] * 0.001), 0.0, 0.0,
                         0.0)),
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
  color = (color / (color + make_float3(1.0)));
  color = powf(color, make_float3((1.0 / 2.2)));
  output.color = make_float4(color, (material.opacity * albedoValue.a));
  output.normalBuffer = make_float4(((worldNormal * 0.5) + 0.5), 1.0);
  output.positionBuffer = make_float4(input.worldPosition, 1.0);
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
  float closestDepth = tex2D(shadowMap, projCoords.xy).r;
  float currentDepth = projCoords.z;
  float bias = fmaxf(
      (0.05 * (1.0 - dot(input.worldNormal, normalize((globals.cameraPosition -
                                                       input.worldPosition))))),
      0.005);
  float shadow = (((currentDepth - bias) > closestDepth) ? 1.0 : 0.0);
  float pcfDepth = 0.0;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 2) texelSize = (1.0 / make_float2(globals.screenSize));
  float offset = (globals.noiseValues[((iteration * 4) % 16)] * 0.001);
  int x = -1;
  for (; (x <= 1); ++x) {
    int y = -1;
    for (; (y <= 1); ++y) {
      float pcfDepth =
          tex2D(shadowMap, ((projCoords.xy + (make_float2(x, y) * texelSize)) +
                            make_float2(offset)))
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
             size = 2) texCoord = make_int2(gl_GlobalInvocationID.xy);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 2) screenSize = globals.screenSize;
  if (((texCoord.x >= int(screenSize.x)) ||
       (texCoord.y >= int(screenSize.y)))) {
    return;
  }
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 2) uv = (make_float2(texCoord) / screenSize);
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) color = make_float4(0.0);
  float totalWeight = 0.0;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 2) direction = (make_float2(0.5) - uv);
  float len = length(direction);
  direction = normalize(direction);
  int i = 0;
  for (; (i < 32); ++i) {
    if ((i >= MAX_ITERATIONS)) {
      break;
    }
    float t = (float(i) / 32.0);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 2) pos = (uv + (((direction * t) * len) * 0.1));
    float noise = fbm(make_float3((pos * 10.0), (globals.scene.time * 0.05)), 4,
                      2.0, 0.5);
    float weight = (1.0 - t);
    weight = (weight * weight);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) noiseColor =
        make_float3(
            (0.5 + (0.5 * sinf((((noise * 5.0) + globals.scene.time) + 0.0)))),
            (0.5 + (0.5 * sinf((((noise * 5.0) + globals.scene.time) + 2.0)))),
            (0.5 + (0.5 * sinf((((noise * 5.0) + globals.scene.time) + 4.0)))));
    color.rgb = (noiseColor * weight);
    totalWeight = weight;
    direction = (mat2(cosf((t * 3.0)), -sinf((t * 3.0)), sinf((t * 3.0)),
                      cosf((t * 3.0))) *
                 direction);
  }
  color.rgb = totalWeight;
  color.a = 1.0;
  float vignette = (1.0 - smoothstep(0.5, 1.0, (length((uv - 0.5)) * 1.5)));
  color.rgb = vignette;
  imageStore(outputImage, texCoord, color);
}
