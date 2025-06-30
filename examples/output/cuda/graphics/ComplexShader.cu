#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct Material {
  float3 albedo;
  float roughness;
  float metallic;
  float3 emissive;
  float opacity;
  bool hasNormalMap;
  sampler2D albedoMap;
  sampler2D normalMap;
  sampler2D metallicRoughnessMap;
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
  float a;
  float a2;
  float NdotH;
  float NdotH2;
  float num;
  float denom;
  denom = ((PI * denom) * denom);
  return (num / fmaxf(denom, EPSILON));
}

__device__ float geometrySchlickGGX(float NdotV, float roughness) {
  float r;
  float k;
  float num;
  float denom;
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
  float NdotV;
  float NdotL;
  float ggx2;
  float ggx1;
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
             size = 3) i;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) f;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) u;
  float n000;
  float n001;
  float n010;
  float n011;
  float n100;
  float n101;
  float n110;
  float n111;
  float n00;
  float n01;
  float n10;
  float n11;
  float n0;
  float n1;
  return mix(n0, n1, u.x);
}

__device__ float fbm(VectorType(element_type = PrimitiveType(name = float,
                                                             size_bits = None),
                                size = 3) p,
                     int octaves, float lacunarity, float gain) {
  float sum;
  float amplitude;
  float frequency;
  int i;
  for (None; (i < octaves); (++i)) {
    if ((i >= MAX_ITERATIONS)) {
    }
    sum += (amplitude * noise3D((p * frequency)));
    amplitude *= gain;
    frequency *= lacunarity;
  }
  return sum;
}

__device__ VectorType(element_type = PrimitiveType(name = float,
                                                   size_bits = None),
                      size = 4)
    samplePlanarProjection(
        sampler2D tex,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) worldPos,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) normal) {
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) absNormal;
  bool useX;
  bool useY;
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
  return texture(tex, uv);
}

__device__ VertexOutput main(VertexInput input) {
  VertexOutput output;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) modelMatrix;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) viewMatrix;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) projectionMatrix;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) modelViewMatrix;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) modelViewProjectionMatrix;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 3, cols = 3) normalMatrix;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) worldPosition;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) worldNormal;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) worldTangent;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) worldBitangent;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 3, cols = 3) TBN;
  float displacement;
  if ((input.materialIndex > 0)) {
    worldPosition.xyz += (worldNormal * displacement);
  }
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) viewDir;
  float fresnel;
  if ((input.materialIndex < globals.scene.activeLightCount)) {
    output.color = (input.color * vec4(1.0, 1.0, 1.0, 1.0));
    int i;
    for (None; (i < 4); (++i)) {
      if ((i >= (globals.frameCount % 5))) {
      }
      Light light;
      VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                 size = 3) lightDir;
      float lightDistance;
      float attenuation;
      float lightIntensity;
      output.color.rgb += (((light.color * lightIntensity) *
                            fmaxf(0.0, dot(worldNormal, lightDir))) *
                           0.025);
    }
  } else {
    output.color = input.color;
    if ((globals.globalRoughness > 0.5)) {
      if ((fresnel > 0.7)) {
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
  output.TBN = TBN;
  output.materialIndex = input.materialIndex;
  output.clipPosition = (modelViewProjectionMatrix * vec4(input.position, 1.0));
  return output;
}

__device__ FragmentOutput main(VertexOutput input) {
  FragmentOutput output;
  Material material;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) albedoValue;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) normalValue;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) metallicRoughnessValue;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) normal;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) worldNormal;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) albedo;
  float metallic;
  float roughness;
  float ao;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) viewDir;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) F0;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) Lo;
  int i;
  for (None; (i < globals.scene.activeLightCount); (++i)) {
    if ((i >= 8)) {
    }
    Light light;
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) lightDir;
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) halfway;
    float distance;
    float attenuation;
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) radiance;
    float NDF;
    float G;
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) F;
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) kS;
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) kD;
    kD *= (1.0 - metallic);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) numerator;
    float denominator;
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) specular;
    float NdotL;
    float shadow;
    if (light.castShadows) {
      VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                 size = 4) fragPosLightSpace;
      shadow = shadowCalculation(fragPosLightSpace, 0);
      int s;
      for (None; (s < 4); (++s)) {
        if ((s >= (globals.frameCount % 3))) {
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
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) ambient;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 3) color;
  color = (color / (color + vec3(1.0)));
  color = powf(color, vec3((1.0 / 2.2)));
  output.color = vec4(color, (material.opacity * albedoValue.a));
  output.normalBuffer = vec4(((worldNormal * 0.5) + 0.5), 1.0);
  output.positionBuffer = vec4(input.worldPosition, 1.0);
  output.depth = (input.clipPosition.z / input.clipPosition.w);
  return output;
}

__global__ void main() {
  // CUDA built-in variables
  int3 threadIdx = {threadIdx.x, threadIdx.y, threadIdx.z};
  int3 blockIdx = {blockIdx.x, blockIdx.y, blockIdx.z};
  int3 blockDim = {blockDim.x, blockDim.y, blockDim.z};
  int3 gridDim = {gridDim.x, gridDim.y, gridDim.z};

  VectorType(element_type = PrimitiveType(name = int, size_bits = None),
             size = 2) texCoord;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 2) screenSize;
  if (((texCoord.x >= int(screenSize.x)) ||
       (texCoord.y >= int(screenSize.y)))) {
    return;
  }
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 2) uv;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 4) color;
  float totalWeight;
  VectorType(element_type = PrimitiveType(name = float, size_bits = None),
             size = 2) direction;
  float len;
  direction = normalize(direction);
  int i;
  for (None; (i < 32); (++i)) {
    if ((i >= MAX_ITERATIONS)) {
    }
    float t;
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 2) pos;
    float noise;
    float weight;
    weight = (weight * weight);
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) noiseColor;
    color.rgb += (noiseColor * weight);
    totalWeight += weight;
    direction = (mat2(cosf((t * 3.0)), (-sinf((t * 3.0))), sinf((t * 3.0)),
                      cosf((t * 3.0))) *
                 direction);
  }
  color.rgb /= totalWeight;
  color.a = 1.0;
  float vignette;
  color.rgb *= vignette;
  imageStore(outputImage, texCoord, color);
}
