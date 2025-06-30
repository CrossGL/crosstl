
#include <metal_stdlib>
using namespace metal;

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
  array<float> noiseValues;
};
float PI;
float EPSILON;
int MAX_ITERATIONS;
float a2;
float num;
float denom;
float geometrySchlickGGX(float NdotV [[stage_in]],
                         float roughness [[stage_in]]) {
  float r = roughness + 1.0;
  float k = r * r / 8.0;
  float num = NdotV;
  float denom = NdotV * 1.0 - k + k;
  return num / max(denom, EPSILON);
}

float geometrySmith(float3 N [[stage_in]], float3 V [[stage_in]],
                    float3 L [[stage_in]], float roughness [[stage_in]]) {
  float NdotV = max(dot(N, V), 0.0);
  float ggx2 = geometrySchlickGGX(NdotV, roughness);
  return ggx1 * ggx2;
}

float3 fresnelSchlick(float cosTheta [[stage_in]], float3 F0 [[stage_in]]) {
  return F0 + 1.0 - F0 * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float noise3D(float3 p [[stage_in]]) {
  float3 i = floor(p);
  float3 u = f * f * f * f * f * 6.0 - 15.0 + 10.0;
  float n000 = fract(sin(dot(i, float3(13.534, 43.5234, 243.32))) * 4453.0);
  float n010 = fract(
      sin(dot(i + float3(0.0, 1.0, 0.0), float3(13.534, 43.5234, 243.32))) *
      4453.0);
  float n100 = fract(
      sin(dot(i + float3(1.0, 0.0, 0.0), float3(13.534, 43.5234, 243.32))) *
      4453.0);
  float n110 = fract(
      sin(dot(i + float3(1.0, 1.0, 0.0), float3(13.534, 43.5234, 243.32))) *
      4453.0);
  float n00 = mix(n000, n001, u.z);
  float n10 = mix(n100, n101, u.z);
  float n0 = mix(n00, n01, u.y);
  return mix(n0, n1, u.x);
}

float fbm(float3 p [[stage_in]], int octaves [[stage_in]],
          float lacunarity [[stage_in]], float gain [[stage_in]]) {
  float sum = 0.0;
  float amplitude = 1.0;
  float frequency = 1.0;
  for (int i = 0; i < octaves; i++) {
    if (i >= MAX_ITERATIONS) {
      break;
    }
    sum += amplitude * noise3D(p * frequency);
    frequency *= lacunarity;
  }
  return sum;
}

float4 samplePlanarProjection(texture2d<float> tex [[stage_in]],
                              float3 worldPos [[stage_in]],
                              float3 normal [[stage_in]]) {
  float3 absNormal = abs(normal);
  bool useY = !useX && absNormal.y >= absNormal.z;
  float2 uv;
  if (useX) {
    uv = worldPos.zy * 0.5 + 0.5;
    if (normal.x < 0.0) {
      uv.x = 1.0 - uv.x;
    }
  } else if (useY) {
    uv = worldPos.xz * 0.5 + 0.5;
    if (normal.y < 0.0) {
      uv.y = 1.0 - uv.y;
    }
  } else {
    uv = worldPos.xy * 0.5 + 0.5;
    if (normal.z < 0.0) {
      uv.x = 1.0 - uv.x;
    }
  }
  return tex.sample(sampler(mag_filter::linear, min_filter::linear), uv);
}
