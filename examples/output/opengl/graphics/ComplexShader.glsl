
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
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) viewProjection;
};
struct Scene {
  Material materials[4];
  Light lights[8];
  vec3 ambientLight;
  float time;
  float elapsedTime;
  int activeLightCount;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) viewMatrix;
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 4, cols = 4) projectionMatrix;
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
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 3, cols = 3) TBN;
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
layout(std140, binding = 3)
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) UP_VECTOR;
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

float noise3D(VectorType(element_type = PrimitiveType(name = float,
                                                      size_bits = None),
                         size = 3) p) {
  vec3 i = IdentifierNode(name = floor)(p);
  vec3 f = IdentifierNode(name = fract)(p);
  vec3 u = (((f * f) * f) * ((f * ((f * 6.0) - 15.0)) + 10.0));
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

float fbm(VectorType(element_type = PrimitiveType(name = float,
                                                  size_bits = None),
                     size = 3) p,
          int octaves, float lacunarity, float gain) {
  float sum = 0.0;
  float amplitude = 1.0;
  float frequency = 1.0;
  for (int i = 0;; (i < octaves); (++i)) {
    if ((i >= MAX_ITERATIONS)) {
    }
    sum += (amplitude * IdentifierNode(name = noise3D)((p * frequency)));
    amplitude *= gain;
    frequency *= lacunarity;
  }
  return sum;
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 4)
    samplePlanarProjection(
        sampler2D tex,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) worldPos,
        VectorType(element_type = PrimitiveType(name = float, size_bits = None),
                   size = 3) normal) {
  vec3 absNormal = IdentifierNode(name = abs)(normal);
  bool useX = ((absNormal.x >= absNormal.y) && (absNormal.x >= absNormal.z));
  bool useY = ((!useX) && (absNormal.y >= absNormal.z));
  vec2 uv;
  if (useX) {
    uv = ((worldPos.zy * 0.5) + 0.5);
    if ((normal.x < 0.0)) {
    }
  } else {
  }
  return IdentifierNode(name = texture)(tex, uv);
}

// Vertex Shader
void main() {
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
  vec4 worldPosition =
      (modelMatrix * IdentifierNode(name = vec4)(input.position, 1.0));
  vec3 worldNormal =
      IdentifierNode(name = normalize)((normalMatrix * input.normal));
  vec3 worldTangent =
      IdentifierNode(name = normalize)((normalMatrix * input.tangent));
  vec3 worldBitangent =
      IdentifierNode(name = normalize)((normalMatrix * input.bitangent));
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 3, cols = 3) TBN =
      IdentifierNode(name = mat3)(worldTangent, worldBitangent, worldNormal);
  float displacement =
      (IdentifierNode(name = fbm)(
           (worldPosition.xyz + (globals.scene.time * 0.1)), 4, 2.0, 0.5) *
       0.1);
  if ((input.materialIndex > 0)) {
    worldPosition.xyz += (worldNormal * displacement);
  }
  vec3 viewDir = IdentifierNode(name = normalize)(
      (globals.cameraPosition - worldPosition.xyz));
  float fresnel = IdentifierNode(name = pow)(
      (1.0 - IdentifierNode(name = max)(
                 0.0, IdentifierNode(name = dot)(worldNormal, viewDir))),
      5.0);
  if ((input.materialIndex < globals.scene.activeLightCount)) {
    output.color =
        (input.color * IdentifierNode(name = vec4)(1.0, 1.0, 1.0, 1.0));
    for (int i = 0;; (i < 4); (++i)) {
      if ((i >= (globals.frameCount % 5))) {
      }
      Light light = globals.scene.lights[i];
      vec3 lightDir = IdentifierNode(name = normalize)(
          (light.position - worldPosition.xyz));
      float lightDistance =
          IdentifierNode(name = length)((light.position - worldPosition.xyz));
      float attenuation = (1.0 / (1.0 + (lightDistance * lightDistance)));
      float lightIntensity = (light.intensity * attenuation);
      output.color.rgb +=
          (((light.color * lightIntensity) *
            IdentifierNode(name = max)(
                0.0, IdentifierNode(name = dot)(worldNormal, lightDir))) *
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
  output.clipPosition = (modelViewProjectionMatrix *
                         IdentifierNode(name = vec4)(input.position, 1.0));
  return output;
}

// Fragment Shader
void main() {
  FragmentOutput output;
  Material material = globals.scene.materials[input.materialIndex];
  vec4 albedoValue =
      IdentifierNode(name = texture)(material.albedoMap, input.texCoord0);
  vec4 normalValue =
      IdentifierNode(name = texture)(material.normalMap, input.texCoord0);
  vec4 metallicRoughnessValue = IdentifierNode(name = texture)(
      material.metallicRoughnessMap, input.texCoord0);
  vec3 normal = ((normalValue.xyz * 2.0) - 1.0);
  vec3 worldNormal = IdentifierNode(name = normalize)((input.TBN * normal));
  vec3 albedo = (albedoValue.rgb * material.albedo);
  float metallic = (metallicRoughnessValue.b * material.metallic);
  float roughness = (metallicRoughnessValue.g * material.roughness);
  float ao = metallicRoughnessValue.r;
  vec3 viewDir = IdentifierNode(name = normalize)(
      (globals.cameraPosition - input.worldPosition));
  vec3 F0 = IdentifierNode(name = mix)(IdentifierNode(name = vec3)(0.04),
                                       albedo, metallic);
  vec3 Lo = IdentifierNode(name = vec3)(0.0);
  for (int i = 0;; (i < globals.scene.activeLightCount); (++i)) {
    if ((i >= 8)) {
    }
    Light light = globals.scene.lights[i];
    vec3 lightDir = IdentifierNode(name = normalize)(
        (light.position - input.worldPosition));
    vec3 halfway = IdentifierNode(name = normalize)((viewDir + lightDir));
    float distance =
        IdentifierNode(name = length)((light.position - input.worldPosition));
    float attenuation = (1.0 / (distance * distance));
    vec3 radiance = ((light.color * light.intensity) * attenuation);
    float NDF =
        IdentifierNode(name = distributionGGX)(worldNormal, halfway, roughness);
    float G = IdentifierNode(name = geometrySmith)(worldNormal, viewDir,
                                                   lightDir, roughness);
    vec3 F = IdentifierNode(name = fresnelSchlick)(
        IdentifierNode(name = max)(IdentifierNode(name = dot)(halfway, viewDir),
                                   0.0),
        F0);
    vec3 kS = F;
    vec3 kD = (IdentifierNode(name = vec3)(1.0) - kS);
    kD *= (1.0 - metallic);
    vec3 numerator = ((NDF * G) * F);
    float denominator =
        (((4.0 * IdentifierNode(name = max)(
                     IdentifierNode(name = dot)(worldNormal, viewDir), 0.0)) *
          IdentifierNode(name = max)(
              IdentifierNode(name = dot)(worldNormal, lightDir), 0.0)) +
         EPSILON);
    vec3 specular = (numerator / denominator);
    float NdotL = IdentifierNode(name = max)(
        IdentifierNode(name = dot)(worldNormal, lightDir), 0.0);
    float shadow = 0.0;
    if (light.castShadows) {
      vec4 fragPosLightSpace =
          (light.viewProjection *
           IdentifierNode(name = vec4)(input.worldPosition, 1.0));
      shadow = IdentifierNode(name = shadowCalculation)(fragPosLightSpace, 0);
      for (int s = 0;; (s < 4); (++s)) {
        if ((s >= (globals.frameCount % 3))) {
        }
        shadow += IdentifierNode(name = shadowCalculation)(
            (fragPosLightSpace +
             IdentifierNode(name = vec4)(
                 (globals.noiseValues[(s % 16)] * 0.001), 0.0, 0.0, 0.0)),
            (s + 1));
      }
      shadow /= 5.0;
    }
    Lo += ((((1.0 - shadow) * (((kD * albedo) / PI) + specular)) * radiance) *
           NdotL);
  }
  vec3 ambient = ((globals.scene.ambientLight * albedo) * ao);
  vec3 color = (ambient + Lo);
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

float shadowCalculation(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 4) fragPosLightSpace,
    int iteration) {}

float shadowCalculation(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 4) fragPosLightSpace,
    int iteration) {
  if ((iteration > 3)) {
  }
  vec3 projCoords = (fragPosLightSpace.xyz / fragPosLightSpace.w);
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
  vec2 texelSize = (1.0 / IdentifierNode(name = vec2)(globals.screenSize));
  float offset = (globals.noiseValues[((iteration * 4) % 16)] * 0.001);
  for (int x = (-1);; (x <= 1); (++x)) {
    for (int y = (-1);; (y <= 1); (++y)) {
      float pcfDepth =
          IdentifierNode(name = texture)(
              shadowMap, ((projCoords.xy +
                           (IdentifierNode(name = vec2)(x, y) * texelSize)) +
                          IdentifierNode(name = vec2)(offset)))
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

// Compute Shader
void main() {
  ivec2 texCoord = IdentifierNode(name = ivec2)(gl_GlobalInvocationID.xy);
  vec2 screenSize = globals.screenSize;
  if (((texCoord.x >= IdentifierNode(name = int)(screenSize.x)) ||
       (texCoord.y >= IdentifierNode(name = int)(screenSize.y)))) {
    return None;
  }
  vec2 uv = (IdentifierNode(name = vec2)(texCoord) / screenSize);
  vec4 color = IdentifierNode(name = vec4)(0.0);
  float totalWeight = 0.0;
  vec2 direction = (IdentifierNode(name = vec2)(0.5) - uv);
  float len = IdentifierNode(name = length)(direction);
  direction = IdentifierNode(name = normalize)(direction);
  for (int i = 0;; (i < 32); (++i)) {
    if ((i >= MAX_ITERATIONS)) {
    }
    float t = (IdentifierNode(name = float)(i) / 32.0);
    vec2 pos = (uv + (((direction * t) * len) * 0.1));
    float noise = IdentifierNode(name = fbm)(
        IdentifierNode(name = vec3)((pos * 10.0), (globals.scene.time * 0.05)),
        4, 2.0, 0.5);
    float weight = (1.0 - t);
    weight = (weight * weight);
    vec3 noiseColor = IdentifierNode(name = vec3)(
        (0.5 + (0.5 * IdentifierNode(name = sin)(
                          (((noise * 5.0) + globals.scene.time) + 0.0)))),
        (0.5 + (0.5 * IdentifierNode(name = sin)(
                          (((noise * 5.0) + globals.scene.time) + 2.0)))),
        (0.5 + (0.5 * IdentifierNode(name = sin)(
                          (((noise * 5.0) + globals.scene.time) + 4.0)))));
    color.rgb += (noiseColor * weight);
    totalWeight += weight;
    direction =
        (IdentifierNode(name = mat2)(IdentifierNode(name = cos)((t * 3.0)),
                                     (-IdentifierNode(name = sin)((t * 3.0))),
                                     IdentifierNode(name = sin)((t * 3.0)),
                                     IdentifierNode(name = cos)((t * 3.0))) *
         direction);
  }
  color.rgb /= totalWeight;
  color.a = 1.0;
  float vignette =
      (1.0 - IdentifierNode(name = smoothstep)(
                 0.5, 1.0, (IdentifierNode(name = length)((uv - 0.5)) * 1.5)));
  color.rgb *= vignette;
  IdentifierNode(name = imageStore)(outputImage, texCoord, color);
}
