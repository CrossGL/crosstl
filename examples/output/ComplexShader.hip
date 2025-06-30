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
  float4x4 viewProjection;
};

struct Scene {
  float3 ambientLight;
  float time;
  float elapsedTime;
  int activeLightCount;
  float4x4 viewMatrix;
  float4x4 projectionMatrix;
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
  float3x3 TBN;
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
};

float PI;
float EPSILON;
int MAX_ITERATIONS;
float a2;
float num;
float denom;
__device__ float geometrySchlickGGX(float NdotV, float roughness) {
  float r;
  float k;
  float num;
  float denom;
  return;
}

__device__ float geometrySmith(float3 N, float3 V, float3 L, float roughness) {
  float NdotV;
  float ggx2;
  return;
}

__device__ float3 fresnelSchlick(float cosTheta, float3 F0) { return; }

__device__ float noise3D(float3 p) {
  float3 i;
  float3 u;
  float n000;
  float n010;
  float n100;
  float n110;
  float n00;
  float n10;
  float n0;
  return;
}

__device__ float fbm(float3 p, int octaves, float lacunarity, float gain) {
  float sum;
  float amplitude;
  float frequency;
  int i;
  i;
  i++;
  for ((= 0); (< octaves);) {
    if ((i >= MAX_ITERATIONS)) {
      break;
    }
    sum;
    frequency;
  }
  return;
}

__device__ float4 samplePlanarProjection(texture<float4, 2> tex,
                                         float3 worldPos, float3 normal) {
  float3 absNormal;
  bool useY;
  float2 uv;
  if (useX) {
    uv;
    if ((normal.x < 0.0)) {
      MemberAccessNode(object = uv, member = x);
    }
  } else {
    uv;
    if ((normal.z < 0.0)) {
      MemberAccessNode(object = uv, member = x);
    }
  }
  return;
}
