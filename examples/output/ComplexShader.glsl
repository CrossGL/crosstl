
#version 450 core
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
struct VertexInput {
  vec3 position;
  vec3 normal;
  vec3 tangent;
  vec3 bitangent;
  vec2 texCoord0;
  vec2 texCoord1;
  vec4 color;
  int materialIndex;
};
struct VertexOutput {
  vec3 worldPosition;
  vec3 worldNormal;
  vec3 worldTangent;
  vec3 worldBitangent;
  vec2 texCoord0;
  vec2 texCoord1;
  vec4 color;
  mat3 TBN;
  int materialIndex;
  vec4 clipPosition;
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
  float noiseValues[];
};
layout(std140, binding = 0) float PI;
layout(std140, binding = 1) float EPSILON;
layout(std140, binding = 2) int MAX_ITERATIONS;
layout(std140, binding = 3) float a2;
layout(std140, binding = 4) float num;
layout(std140, binding = 5) float denom;
float geometrySchlickGGX(float NdotV, float roughness) {
  float r = roughness + 1.0;
  float k = r * r / 8.0;
  float num = NdotV;
  float denom = NdotV * 1.0 - k + k;
  return num / max(denom, EPSILON);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
  float NdotV = max(dot(N, V), 0.0);
  float ggx2 = geometrySchlickGGX(NdotV, roughness);
  return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
  return F0 + 1.0 - F0 * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float noise3D(vec3 p) {
  vec3 i = floor(p);
  vec3 u = f * f * f * f * f * 6.0 - 15.0 + 10.0;
  float n000 = fract(sin(dot(i, vec3(13.534, 43.5234, 243.32))) * 4453.0);
  float n010 =
      fract(sin(dot(i + vec3(0.0, 1.0, 0.0), vec3(13.534, 43.5234, 243.32))) *
            4453.0);
  float n100 =
      fract(sin(dot(i + vec3(1.0, 0.0, 0.0), vec3(13.534, 43.5234, 243.32))) *
            4453.0);
  float n110 =
      fract(sin(dot(i + vec3(1.0, 1.0, 0.0), vec3(13.534, 43.5234, 243.32))) *
            4453.0);
  float n00 = mix(n000, n001, u.z);
  float n10 = mix(n100, n101, u.z);
  float n0 = mix(n00, n01, u.y);
  return mix(n0, n1, u.x);
}

float fbm(vec3 p, int octaves, float lacunarity, float gain) {
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

vec4 samplePlanarProjection(sampler2D tex, vec3 worldPos, vec3 normal) {
  vec3 absNormal = abs(normal);
  bool useY = !useX && absNormal.y >= absNormal.z;
  vec2 uv;
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
  return texture(tex, uv);
}
