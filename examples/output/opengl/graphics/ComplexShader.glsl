
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
float distributionGGX(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) N,
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) H,
    float roughness) {
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

float geometrySmith(
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) N,
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) V,
    VectorType(element_type = PrimitiveType(name = float, size_bits = None),
               size = 3) L,
    float roughness) {
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float ggx2 = geometrySchlickGGX(NdotV, roughness);
  float ggx1 = geometrySchlickGGX(NdotL, roughness);
  return (ggx1 * ggx2);
}

VectorType(element_type = PrimitiveType(name = float, size_bits = None),
           size = 3)
    fresnelSchlick(float cosTheta,
                   VectorType(element_type = PrimitiveType(name = float,
                                                           size_bits = None),
                              size = 3) F0) {
  return (F0 + ((1.0 - F0) * pow(max((1.0 - cosTheta), 0.0), 5.0)));
}

float noise3D(VectorType(element_type = PrimitiveType(name = float,
                                                      size_bits = None),
                         size = 3) p) {
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
    sum += (amplitude * noise3D((p * frequency)));
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
  vec3 absNormal = abs(normal);
  bool useX = ((absNormal.x >= absNormal.y) && (absNormal.x >= absNormal.z));
  bool useY = ((!useX) && (absNormal.y >= absNormal.z));
  vec2 uv;
  if (useX) {
    uv = ((worldPos.zy * 0.5) + 0.5);
    if ((normal.x < 0.0)) {
    }
  } else {
  }
  return texture(tex, uv);
}

// Vertex Shader
void main() {
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
  vec4 worldPosition = (modelMatrix * vec4(input.position, 1.0));
  vec3 worldNormal = normalize((normalMatrix * input.normal));
  vec3 worldTangent = normalize((normalMatrix * input.tangent));
  vec3 worldBitangent = normalize((normalMatrix * input.bitangent));
  MatrixType(element_type = PrimitiveType(name = float, size_bits = None),
             rows = 3, cols = 3) TBN =
      mat3(worldTangent, worldBitangent, worldNormal);
  float displacement =
      (fbm((worldPosition.xyz + (globals.scene.time * 0.1)), 4, 2.0, 0.5) *
       0.1);
  if ((input.materialIndex > 0)) {
    worldPosition.xyz += (worldNormal * displacement);
  }
  vec3 viewDir = normalize((globals.cameraPosition - worldPosition.xyz));
  float fresnel = pow((1.0 - max(0.0, dot(worldNormal, viewDir))), 5.0);
  if ((input.materialIndex < globals.scene.activeLightCount)) {
    output.color = (input.color * vec4(1.0, 1.0, 1.0, 1.0));
    for (int i = 0;; (i < 4); (++i)) {
      if ((i >= (globals.frameCount % 5))) {
      }
      Light light = globals.scene.lights[i];
      vec3 lightDir = normalize((light.position - worldPosition.xyz));
      float lightDistance = length((light.position - worldPosition.xyz));
      float attenuation = (1.0 / (1.0 + (lightDistance * lightDistance)));
      float lightIntensity = (light.intensity * attenuation);
      output.color.rgb += (((light.color * lightIntensity) *
                            max(0.0, dot(worldNormal, lightDir))) *
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

// Fragment Shader
void main() {
  FragmentOutput output;
  Material material = globals.scene.materials[input.materialIndex];
  vec4 albedoValue = texture(material.albedoMap, input.texCoord0);
  vec4 normalValue = texture(material.normalMap, input.texCoord0);
  vec4 metallicRoughnessValue =
      texture(material.metallicRoughnessMap, input.texCoord0);
  vec3 normal = ((normalValue.xyz * 2.0) - 1.0);
  vec3 worldNormal = normalize((input.TBN * normal));
  vec3 albedo = (albedoValue.rgb * material.albedo);
  float metallic = (metallicRoughnessValue.b * material.metallic);
  float roughness = (metallicRoughnessValue.g * material.roughness);
  float ao = metallicRoughnessValue.r;
  vec3 viewDir = normalize((globals.cameraPosition - input.worldPosition));
  vec3 F0 = mix(vec3(0.04), albedo, metallic);
  vec3 Lo = vec3(0.0);
  for (int i = 0;; (i < globals.scene.activeLightCount); (++i)) {
    if ((i >= 8)) {
    }
    Light light = globals.scene.lights[i];
    vec3 lightDir = normalize((light.position - input.worldPosition));
    vec3 halfway = normalize((viewDir + lightDir));
    float distance = length((light.position - input.worldPosition));
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
          (light.viewProjection * vec4(input.worldPosition, 1.0));
      shadow = shadowCalculation(fragPosLightSpace, 0);
      for (int s = 0;; (s < 4); (++s)) {
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
  vec3 ambient = ((globals.scene.ambientLight * albedo) * ao);
  vec3 color = (ambient + Lo);
  color = (color / (color + vec3(1.0)));
  color = pow(color, vec3((1.0 / 2.2)));
  output.color = vec4(color, (material.opacity * albedoValue.a));
  output.normalBuffer = vec4(((worldNormal * 0.5) + 0.5), 1.0);
  output.positionBuffer = vec4(input.worldPosition, 1.0);
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
  float closestDepth = texture(shadowMap, projCoords.xy).r;
  float currentDepth = projCoords.z;
  float bias = max(
      (0.05 * (1.0 - dot(input.worldNormal, normalize((globals.cameraPosition -
                                                       input.worldPosition))))),
      0.005);
  float shadow = (((currentDepth - bias) > closestDepth) ? 1.0 : 0.0);
  float pcfDepth = 0.0;
  vec2 texelSize = (1.0 / vec2(globals.screenSize));
  float offset = (globals.noiseValues[((iteration * 4) % 16)] * 0.001);
  for (int x = (-1);; (x <= 1); (++x)) {
    for (int y = (-1);; (y <= 1); (++y)) {
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

// Compute Shader
void main() {
  ivec2 texCoord = ivec2(gl_GlobalInvocationID.xy);
  vec2 screenSize = globals.screenSize;
  if (((texCoord.x >= int(screenSize.x)) ||
       (texCoord.y >= int(screenSize.y)))) {
    return None;
  }
  vec2 uv = (vec2(texCoord) / screenSize);
  vec4 color = vec4(0.0);
  float totalWeight = 0.0;
  vec2 direction = (vec2(0.5) - uv);
  float len = length(direction);
  direction = normalize(direction);
  for (int i = 0;; (i < 32); (++i)) {
    if ((i >= MAX_ITERATIONS)) {
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
