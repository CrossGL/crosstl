// Generated Rust GPU Shader Code
use gpu::*;
use math::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Material {
  pub albedo : Vec3<f32>,
               pub roughness : f32,
                               pub metallic : f32,
                                              pub emissive : Vec3<f32>,
                                                             pub opacity
      : f32,
        pub hasNormalMap : bool,
                           pub albedoMap : Texture2D<f32>,
                                           pub normalMap
      : Texture2D<f32>,
        pub metallicRoughnessMap : Texture2D<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Light {
  pub position : Vec3<f32>,
                 pub color : Vec3<f32>,
                             pub intensity : f32,
                                             pub radius : f32,
                                                          pub castShadows
      : bool,
        pub viewProjection : mat4x4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Scene {
  pub materials
      : MaterialLiteralNode(value = 4,
                            literal_type = PrimitiveType(name = int,
                                                         size_bits = None)),
        pub lights
      : LightLiteralNode(value = 8,
                         literal_type = PrimitiveType(name = int,
                                                      size_bits = None)),
        pub ambientLight : Vec3<f32>,
                           pub time : f32,
                                      pub elapsedTime : f32,
                                                        pub activeLightCount
      : i32,
        pub viewMatrix : mat4x4,
                         pub projectionMatrix : mat4x4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexInput {
  pub position : Vec3<f32>,
                 pub normal : Vec3<f32>,
                              pub tangent : Vec3<f32>,
                                            pub bitangent : Vec3<f32>,
                                                            pub texCoord0
      : Vec2<f32>,
        pub texCoord1 : Vec2<f32>,
                        pub color : Vec4<f32>,
                                    pub materialIndex : i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexOutput {
  pub worldPosition : Vec3<f32>,
                      pub worldNormal : Vec3<f32>,
                                        pub worldTangent : Vec3<f32>,
                                                           pub worldBitangent
      : Vec3<f32>,
        pub texCoord0 : Vec2<f32>,
                        pub texCoord1 : Vec2<f32>,
                                        pub color : Vec4<f32>,
                                                    pub TBN : mat3x3,
                                                              pub materialIndex
      : i32,
        pub clipPosition : Vec4<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FragmentOutput {
  pub color : Vec4<f32>,
              pub normalBuffer : Vec4<f32>,
                                 pub positionBuffer : Vec4<f32>,
                                                      pub depth : f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GlobalUniforms {
  pub scene : Scene,
              pub cameraPosition : Vec3<f32>,
                                   pub globalRoughness : f32,
                                                         pub screenSize
      : Vec2<f32>,
        pub nearPlane : f32,
                        pub farPlane : f32,
                                       pub frameCount : i32,
                                                        pub noiseValues
      : vecNone,
}

static PI : f32 = Default::default();
static EPSILON : f32 = Default::default();
static MAX_ITERATIONS : i32 = Default::default();
static UP_VECTOR : Vec3<f32> = Default::default();
pub fn distributionGGX(N : Vec3<f32>, H : Vec3<f32>, roughness : f32) -> f32 {
  let mut a : f32 = (roughness * roughness);
  let mut a2 : f32 = (a * a);
  let mut NdotH : f32 = max(dot(N, H), 0.0);
  let mut NdotH2 : f32 = (NdotH * NdotH);
  let mut num : f32 = a2;
  let mut denom : f32 = ((NdotH2 * (a2 - 1.0)) + 1.0);
  denom = ((PI * denom) * denom);
  return (num / max(denom, EPSILON));
}

pub fn geometrySchlickGGX(NdotV : f32, roughness : f32) -> f32 {
  let mut r : f32 = (roughness + 1.0);
  let mut k : f32 = ((r * r) / 8.0);
  let mut num : f32 = NdotV;
  let mut denom : f32 = ((NdotV * (1.0 - k)) + k);
  return (num / max(denom, EPSILON));
}

pub fn geometrySmith(N : Vec3<f32>, V : Vec3<f32>, L : Vec3<f32>,
                     roughness : f32) -> f32 {
  let mut NdotV : f32 = max(dot(N, V), 0.0);
  let mut NdotL : f32 = max(dot(N, L), 0.0);
  let mut ggx2 : f32 = geometrySchlickGGX(NdotV, roughness);
  let mut ggx1 : f32 = geometrySchlickGGX(NdotL, roughness);
  return (ggx1 * ggx2);
}

pub fn fresnelSchlick(cosTheta : f32, F0 : Vec3<f32>) -> Vec3<f32> {
  return (F0 + ((1.0 - F0) * pow(max((1.0 - cosTheta), 0.0), 5.0)));
}

pub fn noise3D(p : Vec3<f32>) -> f32 {
  let mut i : Vec3<f32> = floor(p);
  let mut f : Vec3<f32> = fract(p);
  let mut u : Vec3<f32> = (((f * f) * f) * ((f * ((f * 6.0) - 15.0)) + 10.0));
  let mut n000
      : f32 = fract(
            (sin(dot(i, Vec3<f32>::new (13.534, 43.5234, 243.32))) * 4453.0));
  let mut n001 : f32 =
                     fract((sin(dot((i + Vec3<f32>::new (0.0, 0.0, 1.0)),
                                    Vec3<f32>::new (13.534, 43.5234, 243.32))) *
                            4453.0));
  let mut n010 : f32 =
                     fract((sin(dot((i + Vec3<f32>::new (0.0, 1.0, 0.0)),
                                    Vec3<f32>::new (13.534, 43.5234, 243.32))) *
                            4453.0));
  let mut n011 : f32 =
                     fract((sin(dot((i + Vec3<f32>::new (0.0, 1.0, 1.0)),
                                    Vec3<f32>::new (13.534, 43.5234, 243.32))) *
                            4453.0));
  let mut n100 : f32 =
                     fract((sin(dot((i + Vec3<f32>::new (1.0, 0.0, 0.0)),
                                    Vec3<f32>::new (13.534, 43.5234, 243.32))) *
                            4453.0));
  let mut n101 : f32 =
                     fract((sin(dot((i + Vec3<f32>::new (1.0, 0.0, 1.0)),
                                    Vec3<f32>::new (13.534, 43.5234, 243.32))) *
                            4453.0));
  let mut n110 : f32 =
                     fract((sin(dot((i + Vec3<f32>::new (1.0, 1.0, 0.0)),
                                    Vec3<f32>::new (13.534, 43.5234, 243.32))) *
                            4453.0));
  let mut n111 : f32 =
                     fract((sin(dot((i + Vec3<f32>::new (1.0, 1.0, 1.0)),
                                    Vec3<f32>::new (13.534, 43.5234, 243.32))) *
                            4453.0));
  let mut n00 : f32 = lerp(n000, n001, u.z);
  let mut n01 : f32 = lerp(n010, n011, u.z);
  let mut n10 : f32 = lerp(n100, n101, u.z);
  let mut n11 : f32 = lerp(n110, n111, u.z);
  let mut n0 : f32 = lerp(n00, n01, u.y);
  let mut n1 : f32 = lerp(n10, n11, u.y);
  return lerp(n0, n1, u.x);
}

pub fn fbm(p : Vec3<f32>, octaves : i32, lacunarity : f32, gain : f32) -> f32 {
  let mut sum : f32 = 0.0;
  let mut amplitude : f32 = 1.0;
  let mut frequency : f32 = 1.0;
  let mut i : i32 = 0;
  ;
  while (i < octaves) {
    if (i >= MAX_ITERATIONS) {
    }
    sum += (amplitude * noise3D((p * frequency)));
    amplitude *= gain;
    frequency *= lacunarity;
    (++i);
  }
  return sum;
}

pub fn samplePlanarProjection(tex : Texture2D<f32>, worldPos : Vec3<f32>,
                              normal : Vec3<f32>) -> Vec4<f32> {
  let mut absNormal : Vec3<f32> = abs(normal);
  let mut useX
      : bool = ((absNormal.x >= absNormal.y) && (absNormal.x >= absNormal.z));
  let mut useY : bool = ((!useX) && (absNormal.y >= absNormal.z));
  let mut uv : Vec2<f32>;
  if useX {
    uv = ((worldPos.zy * 0.5) + 0.5);
    if (normal.x < 0.0) {
    }
  } else if useY {
    uv = ((worldPos.xz * 0.5) + 0.5);
    if (normal.y < 0.0) {
    }
  } else {
    uv = ((worldPos.xy * 0.5) + 0.5);
    if (normal.z < 0.0) {
    }
  }
  return sample(tex, uv);
}

// Vertex Shader
#[vertex_shader]
pub fn main(input : VertexInput) -> VertexOutput {
  let mut output : VertexOutput;
  let mut modelMatrix : mat4x4 = Mat4<f32>::new (1.0);
  let mut viewMatrix : mat4x4 = globals.scene.viewMatrix;
  let mut projectionMatrix : mat4x4 = globals.scene.projectionMatrix;
  let mut modelViewMatrix : mat4x4 = (viewMatrix * modelMatrix);
  let mut modelViewProjectionMatrix : mat4x4 =
                                          (projectionMatrix * modelViewMatrix);
  let mut normalMatrix : mat3x3 =
                             Mat3<f32>::new (transpose(inverse(modelMatrix)));
  let mut worldPosition
      : Vec4<f32> = (modelMatrix * Vec4<f32>::new (input.position, 1.0));
  let mut worldNormal : Vec3<f32> = normalize((normalMatrix * input.normal));
  let mut worldTangent : Vec3<f32> = normalize((normalMatrix * input.tangent));
  let mut worldBitangent : Vec3<f32> =
                               normalize((normalMatrix * input.bitangent));
  let mut TBN : mat3x3 =
                    Mat3<f32>::new (worldTangent, worldBitangent, worldNormal);
  let mut displacement
      : f32 = (fbm((worldPosition.xyz + (globals.scene.time * 0.1)), 4, 2.0,
                   0.5) *
               0.1);
  if (input.materialIndex > 0) {
    worldPosition.xyz += (worldNormal * displacement);
  }
  let mut viewDir : Vec3<f32> =
                        normalize((globals.cameraPosition - worldPosition.xyz));
  let mut fresnel : f32 = pow((1.0 - max(0.0, dot(worldNormal, viewDir))), 5.0);
  if (input.materialIndex < globals.scene.activeLightCount) {
    output.color = (input.color * Vec4<f32>::new (1.0, 1.0, 1.0, 1.0));
    let mut i : i32 = 0;
    ;
    while (i < 4) {
      if (i >= (globals.frameCount % 5)) {
      }
      let mut light : Light = globals.scene.lights[i];
      let mut lightDir : Vec3<f32> =
                             normalize((light.position - worldPosition.xyz));
      let mut lightDistance : f32 =
                                  length((light.position - worldPosition.xyz));
      let mut attenuation : f32 =
                                (1.0 / (1.0 + (lightDistance * lightDistance)));
      let mut lightIntensity : f32 = (light.intensity * attenuation);
      output.color.rgb += (((light.color * lightIntensity) *
                            max(0.0, dot(worldNormal, lightDir))) *
                           0.025);
      (++i);
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
  output.TBN = TBN;
  output.materialIndex = input.materialIndex;
  output.clipPosition =
      (modelViewProjectionMatrix * Vec4<f32>::new (input.position, 1.0));
  return output;
}

// Fragment Shader
#[fragment_shader]
pub fn main(input : VertexOutput) -> FragmentOutput {
  let mut output : FragmentOutput;
  let mut material : Material = globals.scene.materials[input.materialIndex];
  let mut albedoValue : Vec4<f32> = sample(material.albedoMap, input.texCoord0);
  let mut normalValue : Vec4<f32> = sample(material.normalMap, input.texCoord0);
  let mut metallicRoughnessValue
      : Vec4<f32> = sample(material.metallicRoughnessMap, input.texCoord0);
  let mut normal : Vec3<f32> = ((normalValue.xyz * 2.0) - 1.0);
  let mut worldNormal : Vec3<f32> = normalize((input.TBN * normal));
  let mut albedo : Vec3<f32> = (albedoValue.rgb * material.albedo);
  let mut metallic : f32 = (metallicRoughnessValue.b * material.metallic);
  let mut roughness : f32 = (metallicRoughnessValue.g * material.roughness);
  let mut ao : f32 = metallicRoughnessValue.r;
  let mut viewDir
      : Vec3<f32> = normalize((globals.cameraPosition - input.worldPosition));
  let mut F0 : Vec3<f32> = lerp(Vec3<f32>::new (0.04), albedo, metallic);
  let mut Lo : Vec3<f32> = Vec3<f32>::new (0.0);
  let mut i : i32 = 0;
  ;
  while (i < globals.scene.activeLightCount) {
    if (i >= 8) {
    }
    let mut light : Light = globals.scene.lights[i];
    let mut lightDir : Vec3<f32> =
                           normalize((light.position - input.worldPosition));
    let mut halfway : Vec3<f32> = normalize((viewDir + lightDir));
    let mut distance : f32 = length((light.position - input.worldPosition));
    let mut attenuation : f32 = (1.0 / (distance * distance));
    let mut radiance : Vec3<f32> =
                           ((light.color * light.intensity) * attenuation);
    let mut NDF : f32 = distributionGGX(worldNormal, halfway, roughness);
    let mut G : f32 = geometrySmith(worldNormal, viewDir, lightDir, roughness);
    let mut F : Vec3<f32> = fresnelSchlick(max(dot(halfway, viewDir), 0.0), F0);
    let mut kS : Vec3<f32> = F;
    let mut kD : Vec3<f32> = (Vec3<f32>::new (1.0) - kS);
    kD *= (1.0 - metallic);
    let mut numerator : Vec3<f32> = ((NDF * G) * F);
    let mut denominator : f32 = (((4.0 * max(dot(worldNormal, viewDir), 0.0)) *
                                  max(dot(worldNormal, lightDir), 0.0)) +
                                 EPSILON);
    let mut specular : Vec3<f32> = (numerator / denominator);
    let mut NdotL : f32 = max(dot(worldNormal, lightDir), 0.0);
    let mut shadow : f32 = 0.0;
    if light
      .castShadows {
        let mut fragPosLightSpace
            : Vec4<f32> = (light.viewProjection *
                           Vec4<f32>::new (input.worldPosition, 1.0));
        shadow = shadowCalculation(fragPosLightSpace, 0);
        let mut s : i32 = 0;
        ;
        while (s < 4) {
          if (s >= (globals.frameCount % 3)) {
          }
          shadow += shadowCalculation(
              (fragPosLightSpace +
               Vec4<f32>::new ((globals.noiseValues[(s % 16)] * 0.001), 0.0,
                               0.0, 0.0)),
              (s + 1));
          (++s);
        }
        shadow /= 5.0;
      }
    Lo += ((((1.0 - shadow) * (((kD * albedo) / PI) + specular)) * radiance) *
           NdotL);
    (++i);
  }
  let mut ambient : Vec3<f32> = ((globals.scene.ambientLight * albedo) * ao);
  let mut color : Vec3<f32> = (ambient + Lo);
  color = (color / (color + Vec3<f32>::new (1.0)));
  color = pow(color, Vec3<f32>::new ((1.0 / 2.2)));
  output.color = Vec4<f32>::new (color, (material.opacity * albedoValue.a));
  output.normalBuffer = Vec4<f32>::new (((worldNormal * 0.5) + 0.5), 1.0);
  output.positionBuffer = Vec4<f32>::new (input.worldPosition, 1.0);
  output.depth = (input.clipPosition.z / input.clipPosition.w);
  return output;
}

pub fn shadowCalculation(fragPosLightSpace : Vec4<f32>, iteration : i32)
    -> f32 {}

pub fn shadowCalculation(fragPosLightSpace : Vec4<f32>, iteration : i32)
    -> f32 {
  if (iteration > 3) {
  }
  let mut projCoords : Vec3<f32> =
                           (fragPosLightSpace.xyz / fragPosLightSpace.w);
  projCoords = ((projCoords * 0.5) + 0.5);
  let mut closestDepth : f32 = sample(shadowMap, projCoords.xy).r;
  let mut currentDepth : f32 = projCoords.z;
  let mut bias : f32 =
                     max((0.05 * (1.0 - dot(input.worldNormal,
                                            normalize((globals.cameraPosition -
                                                       input.worldPosition))))),
                         0.005);
  let mut shadow
      : f32 = (if ((currentDepth - bias) > closestDepth){1.0} else {0.0});
  let mut pcfDepth : f32 = 0.0;
  let mut texelSize : Vec2<f32> = (1.0 / Vec2<f32>::new (globals.screenSize));
  let mut offset : f32 = (globals.noiseValues[((iteration * 4) % 16)] * 0.001);
  let mut x : i32 = (-1);
  ;
  while (x <= 1) {
    let mut y : i32 = (-1);
    ;
    while (y <= 1) {
      let mut pcfDepth
          : f32 =
                sample(shadowMap,
                       ((projCoords.xy + (Vec2<f32>::new (x, y) * texelSize)) +
                        Vec2<f32>::new (offset)))
                    .r;
      shadow += (if ((currentDepth - bias) > pcfDepth){1.0} else {0.0});
      (++y);
    }
    (++x);
  }
  shadow /= 9.0;
  if (projCoords.z > 1.0) {
    shadow = 0.0;
  }
  return shadow;
}

// Compute Shader
#[compute_shader]
pub fn main() -> () {
  let mut texCoord : Vec2<i32> = Vec2<i32>::new (gl_GlobalInvocationID.xy);
  let mut screenSize : Vec2<f32> = globals.screenSize;
  if ((texCoord.x >= int(screenSize.x)) || (texCoord.y >= int(screenSize.y))) {
    return;
  }
  let mut uv : Vec2<f32> = (Vec2<f32>::new (texCoord) / screenSize);
  let mut color : Vec4<f32> = Vec4<f32>::new (0.0);
  let mut totalWeight : f32 = 0.0;
  let mut direction : Vec2<f32> = (Vec2<f32>::new (0.5) - uv);
  let mut len : f32 = length(direction);
  direction = normalize(direction);
  let mut i : i32 = 0;
  ;
  while (i < 32) {
    if (i >= MAX_ITERATIONS) {
    }
    let mut t : f32 = (float(i) / 32.0);
    let mut pos : Vec2<f32> = (uv + (((direction * t) * len) * 0.1));
    let mut noise
        : f32 = fbm(Vec3<f32>::new ((pos * 10.0), (globals.scene.time * 0.05)),
                    4, 2.0, 0.5);
    let mut weight : f32 = (1.0 - t);
    weight = (weight * weight);
    let mut noiseColor
        : Vec3<f32> = Vec3<f32>::new (
              (0.5 + (0.5 * sin((((noise * 5.0) + globals.scene.time) + 0.0)))),
              (0.5 + (0.5 * sin((((noise * 5.0) + globals.scene.time) + 2.0)))),
              (0.5 +
               (0.5 * sin((((noise * 5.0) + globals.scene.time) + 4.0)))));
    color.rgb += (noiseColor * weight);
    totalWeight += weight;
    direction = (Mat2<f32>::new (cos((t * 3.0)), (-sin((t * 3.0))),
                                 sin((t * 3.0)), cos((t * 3.0))) *
                 direction);
    (++i);
  }
  color.rgb /= totalWeight;
  color.a = 1.0;
  let mut vignette
      : f32 = (1.0 - smoothstep(0.5, 1.0, (length((uv - 0.5)) * 1.5)));
  color.rgb *= vignette;
  imageStore(outputImage, texCoord, color);
}
