// Generated Rust GPU Shader Code
use gpu::*;
use math::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Material {
  pub albedo : Vec3<f32>, pub roughness : f32, pub metallic : f32,
      pub emissive : Vec3<f32>, pub opacity : f32, pub hasNormalMap : bool,
      pub albedoMap : Texture2D<f32>, pub normalMap : Texture2D<f32>,
      pub metallicRoughnessMap : Texture2D<f32>,
}

impl Material {
  pub fn new (albedo : Vec3<f32>, roughness : f32, metallic : f32,
              emissive : Vec3<f32>, opacity : f32, hasNormalMap : bool,
              albedoMap : Texture2D<f32>, normalMap : Texture2D<f32>,
              metallicRoughnessMap : Texture2D<f32>)
      ->Self {
    Self {
      albedo, roughness, metallic, emissive, opacity, hasNormalMap, albedoMap,
          normalMap, metallicRoughnessMap
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Light {
  pub position : Vec3<f32>, pub color : Vec3<f32>, pub intensity : f32,
      pub radius : f32, pub castShadows : bool, pub viewProjection : Mat4<f32>,
}

impl Light {
  pub fn new (position : Vec3<f32>, color : Vec3<f32>, intensity : f32,
              radius : f32, castShadows : bool, viewProjection : Mat4<f32>)
      ->Self {
    Self { position, color, intensity, radius, castShadows, viewProjection }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Scene {
  pub materials : [Material; 4], pub lights : [Light; 8],
      pub ambientLight : Vec3<f32>, pub time : f32, pub elapsedTime : f32,
      pub activeLightCount : i32, pub viewMatrix : Mat4<f32>,
      pub projectionMatrix : Mat4<f32>,
}

impl Scene {
  pub fn new (materials : [Material; 4], lights : [Light; 8],
              ambientLight : Vec3<f32>, time : f32, elapsedTime : f32,
              activeLightCount : i32, viewMatrix : Mat4<f32>,
              projectionMatrix : Mat4<f32>)
      ->Self {
    Self {
      materials, lights, ambientLight, time, elapsedTime, activeLightCount,
          viewMatrix, projectionMatrix
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VertexInput {
  pub position : Vec3<f32>, pub normal : Vec3<f32>, pub tangent : Vec3<f32>,
      pub bitangent : Vec3<f32>, pub texCoord0 : Vec2<f32>,
      pub texCoord1 : Vec2<f32>, pub color : Vec4<f32>, pub materialIndex : i32,
}

impl VertexInput {
  pub fn new (position : Vec3<f32>, normal : Vec3<f32>, tangent : Vec3<f32>,
              bitangent : Vec3<f32>, texCoord0 : Vec2<f32>,
              texCoord1 : Vec2<f32>, color : Vec4<f32>, materialIndex : i32)
      ->Self {
    Self {
      position, normal, tangent, bitangent, texCoord0, texCoord1, color,
          materialIndex
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VertexOutput {
  pub worldPosition : Vec3<f32>, pub worldNormal : Vec3<f32>,
      pub worldTangent : Vec3<f32>, pub worldBitangent : Vec3<f32>,
      pub texCoord0 : Vec2<f32>, pub texCoord1 : Vec2<f32>,
      pub color : Vec4<f32>, pub TBN : Mat3<f32>, pub materialIndex : i32,
      pub clipPosition : Vec4<f32>,
}

impl VertexOutput {
  pub fn new (worldPosition : Vec3<f32>, worldNormal : Vec3<f32>,
              worldTangent : Vec3<f32>, worldBitangent : Vec3<f32>,
              texCoord0 : Vec2<f32>, texCoord1 : Vec2<f32>, color : Vec4<f32>,
              TBN : Mat3<f32>, materialIndex : i32, clipPosition : Vec4<f32>)
      ->Self {
    Self {
      worldPosition, worldNormal, worldTangent, worldBitangent, texCoord0,
          texCoord1, color, TBN, materialIndex, clipPosition
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FragmentOutput {
  pub color : Vec4<f32>, pub normalBuffer : Vec4<f32>,
      pub positionBuffer : Vec4<f32>, pub depth : f32,
}

impl FragmentOutput {
  pub fn new (color : Vec4<f32>, normalBuffer : Vec4<f32>,
              positionBuffer : Vec4<f32>, depth : f32)
      ->Self {
    Self { color, normalBuffer, positionBuffer, depth }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct GlobalUniforms {
  pub scene : Scene, pub cameraPosition : Vec3<f32>, pub globalRoughness : f32,
      pub screenSize : Vec2<f32>, pub nearPlane : f32, pub farPlane : f32,
      pub frameCount : i32, pub noiseValues : Vec<f32>,
}

impl GlobalUniforms {
  pub fn new (scene : Scene, cameraPosition : Vec3<f32>, globalRoughness : f32,
              screenSize : Vec2<f32>, nearPlane : f32, farPlane : f32,
              frameCount : i32, noiseValues : Vec<f32>)
      ->Self {
    Self {
      scene, cameraPosition, globalRoughness, screenSize, nearPlane, farPlane,
          frameCount, noiseValues
    }
  }
}

const PI : f32 = 3.14159265359;
const EPSILON : f32 = 0.0001;
const MAX_ITERATIONS : i32 = 64;
static UP_VECTOR
    : std::sync::LazyLock<Vec3<f32>> =
          std::sync::LazyLock::new (|| Vec3::<f32>::new (0.0, 1.0, 0.0));

// Constant Buffers
pub fn distributionGGX(N : Vec3<f32>, H : Vec3<f32>, roughness : f32) -> f32 {
  let a : f32 = (roughness * roughness);
  let a2 : f32 = (a * a);
  let NdotH : f32 = max(dot(N, H), 0.0);
  let NdotH2 : f32 = (NdotH * NdotH);
  let num : f32 = a2;
  let mut denom : f32 = ((NdotH2 * (a2 - 1.0)) + 1.0);
  denom = ((PI * denom) * denom);
  return (num / max(denom, EPSILON));
}

pub fn geometrySchlickGGX(NdotV : f32, roughness : f32) -> f32 {
  let r : f32 = (roughness + 1.0);
  let k : f32 = ((r * r) / 8.0);
  let num : f32 = NdotV;
  let denom : f32 = ((NdotV * (1.0 - k)) + k);
  return (num / max(denom, EPSILON));
}

pub fn geometrySmith(N : Vec3<f32>, V : Vec3<f32>, L : Vec3<f32>,
                     roughness : f32) -> f32 {
  let NdotV : f32 = max(dot(N, V), 0.0);
  let NdotL : f32 = max(dot(N, L), 0.0);
  let ggx2 : f32 = geometrySchlickGGX(NdotV, roughness);
  let ggx1 : f32 = geometrySchlickGGX(NdotL, roughness);
  return (ggx1 * ggx2);
}

pub fn fresnelSchlick(cosTheta : f32, F0 : Vec3<f32>) -> Vec3<f32> {
  return (F0 + {
    let __cgl_vec_arg_0 =
        Vec3::<f32>::new ((1.0 - F0.x), (1.0 - F0.y), (1.0 - F0.z));
    let __cgl_vec_arg_1 = pow(max((1.0 - cosTheta), 0.0), 5.0);
    Vec3::<f32>::new ((__cgl_vec_arg_0.x * __cgl_vec_arg_1),
                      (__cgl_vec_arg_0.y * __cgl_vec_arg_1),
                      (__cgl_vec_arg_0.z * __cgl_vec_arg_1))
  });
}

pub fn noise3D(p : Vec3<f32>) -> f32 {
  let i : Vec3<f32> = floor(p);
  let f : Vec3<f32> = fract(p);
  let u : Vec3<f32> = (((f * f) * f) * {
    let __cgl_vec_arg_3 = (f * {
      let __cgl_vec_arg_2 =
          Vec3::<f32>::new ((f.x * 6.0), (f.y * 6.0), (f.z * 6.0));
      Vec3::<f32>::new ((__cgl_vec_arg_2.x - 15.0), (__cgl_vec_arg_2.y - 15.0),
                        (__cgl_vec_arg_2.z - 15.0))
    });
    Vec3::<f32>::new ((__cgl_vec_arg_3.x + 10.0), (__cgl_vec_arg_3.y + 10.0),
                      (__cgl_vec_arg_3.z + 10.0))
  });
  let n000
      : f32 = fract(
            (sin(dot(i, Vec3::<f32>::new (13.534, 43.5234, 243.32))) * 4453.0));
  let n001 : f32 = fract((sin(dot((i + Vec3::<f32>::new (0.0, 0.0, 1.0)),
                                  Vec3::<f32>::new (13.534, 43.5234, 243.32))) *
                          4453.0));
  let n010 : f32 = fract((sin(dot((i + Vec3::<f32>::new (0.0, 1.0, 0.0)),
                                  Vec3::<f32>::new (13.534, 43.5234, 243.32))) *
                          4453.0));
  let n011 : f32 = fract((sin(dot((i + Vec3::<f32>::new (0.0, 1.0, 1.0)),
                                  Vec3::<f32>::new (13.534, 43.5234, 243.32))) *
                          4453.0));
  let n100 : f32 = fract((sin(dot((i + Vec3::<f32>::new (1.0, 0.0, 0.0)),
                                  Vec3::<f32>::new (13.534, 43.5234, 243.32))) *
                          4453.0));
  let n101 : f32 = fract((sin(dot((i + Vec3::<f32>::new (1.0, 0.0, 1.0)),
                                  Vec3::<f32>::new (13.534, 43.5234, 243.32))) *
                          4453.0));
  let n110 : f32 = fract((sin(dot((i + Vec3::<f32>::new (1.0, 1.0, 0.0)),
                                  Vec3::<f32>::new (13.534, 43.5234, 243.32))) *
                          4453.0));
  let n111 : f32 = fract((sin(dot((i + Vec3::<f32>::new (1.0, 1.0, 1.0)),
                                  Vec3::<f32>::new (13.534, 43.5234, 243.32))) *
                          4453.0));
  let n00 : f32 = lerp(n000, n001, u.z);
  let n01 : f32 = lerp(n010, n011, u.z);
  let n10 : f32 = lerp(n100, n101, u.z);
  let n11 : f32 = lerp(n110, n111, u.z);
  let n0 : f32 = lerp(n00, n01, u.y);
  let n1 : f32 = lerp(n10, n11, u.y);
  return lerp(n0, n1, u.x);
}

pub fn fbm(p : Vec3<f32>, octaves : i32, lacunarity : f32, gain : f32) -> f32 {
  let mut sum : f32 = 0.0;
  let mut amplitude : f32 = 1.0;
  let mut frequency : f32 = 1.0;
  let mut i : i32 = 0;
  while (i < octaves) {
    if (i >= MAX_ITERATIONS) {
    }
    sum += (amplitude *
            noise3D(Vec3::<f32>::new ((p.x * frequency), (p.y * frequency),
                                      (p.z * frequency))));
    amplitude *= gain;
    frequency *= lacunarity;
    i += 1;
  }
  return sum;
}

pub fn samplePlanarProjection(tex : Texture2D<f32>, worldPos : Vec3<f32>,
                              normal : Vec3<f32>) -> Vec4<f32> {
  let absNormal : Vec3<f32> = abs(normal);
  let useX : bool =
                 ((absNormal.x >= absNormal.y) && (absNormal.x >= absNormal.z));
  let useY : bool = (!(useX) && (absNormal.y >= absNormal.z));
  let mut uv : Vec2<f32> = Default::default();
  if useX {
    uv = {
      let __cgl_vec_arg_4 =
          Vec2::<f32>::new ((worldPos.z * 0.5), (worldPos.y * 0.5));
    Vec2::<f32>::new ((__cgl_vec_arg_4.x + 0.5), (__cgl_vec_arg_4.y + 0.5))
  };
  if (normal.x < 0.0) {
  }
}
else if useY {
  uv = {
    let __cgl_vec_arg_5 =
        Vec2::<f32>::new ((worldPos.x * 0.5), (worldPos.z * 0.5));
  Vec2::<f32>::new ((__cgl_vec_arg_5.x + 0.5), (__cgl_vec_arg_5.y + 0.5))
};
if (normal.y < 0.0) {
}
}
else {
  uv = {
    let __cgl_vec_arg_6 =
        Vec2::<f32>::new ((worldPos.x * 0.5), (worldPos.y * 0.5));
  Vec2::<f32>::new ((__cgl_vec_arg_6.x + 0.5), (__cgl_vec_arg_6.y + 0.5))
};
if (normal.z < 0.0) {
}
}
return sample(tex, uv);
}

static GLOBALS : std::sync::LazyLock<GlobalUniforms> =
                     std::sync::LazyLock::new (|| Default::default());
// Vertex Shader
#[cfg_attr(feature = "crossgl_gpu", vertex_shader)]
pub fn vertex_main(input : VertexInput) -> VertexOutput {
  let mut output : VertexOutput = Default::default();
  let modelMatrix
      : Mat4<f32> = Mat4::<f32>::new (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  let viewMatrix : Mat4<f32> = (*GLOBALS).scene.viewMatrix;
  let projectionMatrix : Mat4<f32> = (*GLOBALS).scene.projectionMatrix;
  let modelViewMatrix
      : Mat4<f32> = Mat4::<f32>::new (((((viewMatrix.c0.x * modelMatrix.c0.x) +
                                         (viewMatrix.c1.x * modelMatrix.c0.y)) +
                                        (viewMatrix.c2.x * modelMatrix.c0.z)) +
                                       (viewMatrix.c3.x * modelMatrix.c0.w)),
                                      ((((viewMatrix.c0.y * modelMatrix.c0.x) +
                                         (viewMatrix.c1.y * modelMatrix.c0.y)) +
                                        (viewMatrix.c2.y * modelMatrix.c0.z)) +
                                       (viewMatrix.c3.y * modelMatrix.c0.w)),
                                      ((((viewMatrix.c0.z * modelMatrix.c0.x) +
                                         (viewMatrix.c1.z * modelMatrix.c0.y)) +
                                        (viewMatrix.c2.z * modelMatrix.c0.z)) +
                                       (viewMatrix.c3.z * modelMatrix.c0.w)),
                                      ((((viewMatrix.c0.w * modelMatrix.c0.x) +
                                         (viewMatrix.c1.w * modelMatrix.c0.y)) +
                                        (viewMatrix.c2.w * modelMatrix.c0.z)) +
                                       (viewMatrix.c3.w * modelMatrix.c0.w)),
                                      ((((viewMatrix.c0.x * modelMatrix.c1.x) +
                                         (viewMatrix.c1.x * modelMatrix.c1.y)) +
                                        (viewMatrix.c2.x * modelMatrix.c1.z)) +
                                       (viewMatrix.c3.x * modelMatrix.c1.w)),
                                      ((((viewMatrix.c0.y * modelMatrix.c1.x) +
                                         (viewMatrix.c1.y * modelMatrix.c1.y)) +
                                        (viewMatrix.c2.y * modelMatrix.c1.z)) +
                                       (viewMatrix.c3.y * modelMatrix.c1.w)),
                                      ((((viewMatrix.c0.z * modelMatrix.c1.x) +
                                         (viewMatrix.c1.z * modelMatrix.c1.y)) +
                                        (viewMatrix.c2.z * modelMatrix.c1.z)) +
                                       (viewMatrix.c3.z * modelMatrix.c1.w)),
                                      ((((viewMatrix.c0.w * modelMatrix.c1.x) +
                                         (viewMatrix.c1.w * modelMatrix.c1.y)) +
                                        (viewMatrix.c2.w * modelMatrix.c1.z)) +
                                       (viewMatrix.c3.w * modelMatrix.c1.w)),
                                      ((((viewMatrix.c0.x * modelMatrix.c2.x) +
                                         (viewMatrix.c1.x * modelMatrix.c2.y)) +
                                        (viewMatrix.c2.x * modelMatrix.c2.z)) +
                                       (viewMatrix.c3.x * modelMatrix.c2.w)),
                                      ((((viewMatrix.c0.y * modelMatrix.c2.x) +
                                         (viewMatrix.c1.y * modelMatrix.c2.y)) +
                                        (viewMatrix.c2.y * modelMatrix.c2.z)) +
                                       (viewMatrix.c3.y * modelMatrix.c2.w)),
                                      ((((viewMatrix.c0.z * modelMatrix.c2.x) +
                                         (viewMatrix.c1.z * modelMatrix.c2.y)) +
                                        (viewMatrix.c2.z * modelMatrix.c2.z)) +
                                       (viewMatrix.c3.z * modelMatrix.c2.w)),
                                      ((((viewMatrix.c0.w * modelMatrix.c2.x) +
                                         (viewMatrix.c1.w * modelMatrix.c2.y)) +
                                        (viewMatrix.c2.w * modelMatrix.c2.z)) +
                                       (viewMatrix.c3.w * modelMatrix.c2.w)),
                                      ((((viewMatrix.c0.x * modelMatrix.c3.x) +
                                         (viewMatrix.c1.x * modelMatrix.c3.y)) +
                                        (viewMatrix.c2.x * modelMatrix.c3.z)) +
                                       (viewMatrix.c3.x * modelMatrix.c3.w)),
                                      ((((viewMatrix.c0.y * modelMatrix.c3.x) +
                                         (viewMatrix.c1.y * modelMatrix.c3.y)) +
                                        (viewMatrix.c2.y * modelMatrix.c3.z)) +
                                       (viewMatrix.c3.y * modelMatrix.c3.w)),
                                      ((((viewMatrix.c0.z * modelMatrix.c3.x) +
                                         (viewMatrix.c1.z * modelMatrix.c3.y)) +
                                        (viewMatrix.c2.z * modelMatrix.c3.z)) +
                                       (viewMatrix.c3.z * modelMatrix.c3.w)),
                                      ((((viewMatrix.c0.w * modelMatrix.c3.x) +
                                         (viewMatrix.c1.w * modelMatrix.c3.y)) +
                                        (viewMatrix.c2.w * modelMatrix.c3.z)) +
                                       (viewMatrix.c3.w * modelMatrix.c3.w)));
  let modelViewProjectionMatrix
      : Mat4<f32> = Mat4::<f32>::new (
            ((((projectionMatrix.c0.x * modelViewMatrix.c0.x) +
               (projectionMatrix.c1.x * modelViewMatrix.c0.y)) +
              (projectionMatrix.c2.x * modelViewMatrix.c0.z)) +
             (projectionMatrix.c3.x * modelViewMatrix.c0.w)),
            ((((projectionMatrix.c0.y * modelViewMatrix.c0.x) +
               (projectionMatrix.c1.y * modelViewMatrix.c0.y)) +
              (projectionMatrix.c2.y * modelViewMatrix.c0.z)) +
             (projectionMatrix.c3.y * modelViewMatrix.c0.w)),
            ((((projectionMatrix.c0.z * modelViewMatrix.c0.x) +
               (projectionMatrix.c1.z * modelViewMatrix.c0.y)) +
              (projectionMatrix.c2.z * modelViewMatrix.c0.z)) +
             (projectionMatrix.c3.z * modelViewMatrix.c0.w)),
            ((((projectionMatrix.c0.w * modelViewMatrix.c0.x) +
               (projectionMatrix.c1.w * modelViewMatrix.c0.y)) +
              (projectionMatrix.c2.w * modelViewMatrix.c0.z)) +
             (projectionMatrix.c3.w * modelViewMatrix.c0.w)),
            ((((projectionMatrix.c0.x * modelViewMatrix.c1.x) +
               (projectionMatrix.c1.x * modelViewMatrix.c1.y)) +
              (projectionMatrix.c2.x * modelViewMatrix.c1.z)) +
             (projectionMatrix.c3.x * modelViewMatrix.c1.w)),
            ((((projectionMatrix.c0.y * modelViewMatrix.c1.x) +
               (projectionMatrix.c1.y * modelViewMatrix.c1.y)) +
              (projectionMatrix.c2.y * modelViewMatrix.c1.z)) +
             (projectionMatrix.c3.y * modelViewMatrix.c1.w)),
            ((((projectionMatrix.c0.z * modelViewMatrix.c1.x) +
               (projectionMatrix.c1.z * modelViewMatrix.c1.y)) +
              (projectionMatrix.c2.z * modelViewMatrix.c1.z)) +
             (projectionMatrix.c3.z * modelViewMatrix.c1.w)),
            ((((projectionMatrix.c0.w * modelViewMatrix.c1.x) +
               (projectionMatrix.c1.w * modelViewMatrix.c1.y)) +
              (projectionMatrix.c2.w * modelViewMatrix.c1.z)) +
             (projectionMatrix.c3.w * modelViewMatrix.c1.w)),
            ((((projectionMatrix.c0.x * modelViewMatrix.c2.x) +
               (projectionMatrix.c1.x * modelViewMatrix.c2.y)) +
              (projectionMatrix.c2.x * modelViewMatrix.c2.z)) +
             (projectionMatrix.c3.x * modelViewMatrix.c2.w)),
            ((((projectionMatrix.c0.y * modelViewMatrix.c2.x) +
               (projectionMatrix.c1.y * modelViewMatrix.c2.y)) +
              (projectionMatrix.c2.y * modelViewMatrix.c2.z)) +
             (projectionMatrix.c3.y * modelViewMatrix.c2.w)),
            ((((projectionMatrix.c0.z * modelViewMatrix.c2.x) +
               (projectionMatrix.c1.z * modelViewMatrix.c2.y)) +
              (projectionMatrix.c2.z * modelViewMatrix.c2.z)) +
             (projectionMatrix.c3.z * modelViewMatrix.c2.w)),
            ((((projectionMatrix.c0.w * modelViewMatrix.c2.x) +
               (projectionMatrix.c1.w * modelViewMatrix.c2.y)) +
              (projectionMatrix.c2.w * modelViewMatrix.c2.z)) +
             (projectionMatrix.c3.w * modelViewMatrix.c2.w)),
            ((((projectionMatrix.c0.x * modelViewMatrix.c3.x) +
               (projectionMatrix.c1.x * modelViewMatrix.c3.y)) +
              (projectionMatrix.c2.x * modelViewMatrix.c3.z)) +
             (projectionMatrix.c3.x * modelViewMatrix.c3.w)),
            ((((projectionMatrix.c0.y * modelViewMatrix.c3.x) +
               (projectionMatrix.c1.y * modelViewMatrix.c3.y)) +
              (projectionMatrix.c2.y * modelViewMatrix.c3.z)) +
             (projectionMatrix.c3.y * modelViewMatrix.c3.w)),
            ((((projectionMatrix.c0.z * modelViewMatrix.c3.x) +
               (projectionMatrix.c1.z * modelViewMatrix.c3.y)) +
              (projectionMatrix.c2.z * modelViewMatrix.c3.z)) +
             (projectionMatrix.c3.z * modelViewMatrix.c3.w)),
            ((((projectionMatrix.c0.w * modelViewMatrix.c3.x) +
               (projectionMatrix.c1.w * modelViewMatrix.c3.y)) +
              (projectionMatrix.c2.w * modelViewMatrix.c3.z)) +
             (projectionMatrix.c3.w * modelViewMatrix.c3.w)));
  let normalMatrix : Mat3<f32> = {
    let __cgl_mat_arg_0 = transpose(inverse(modelMatrix));
  Mat3::<f32>::new (
      __cgl_mat_arg_0.c0.x, __cgl_mat_arg_0.c0.y, __cgl_mat_arg_0.c0.z,
      __cgl_mat_arg_0.c1.x, __cgl_mat_arg_0.c1.y, __cgl_mat_arg_0.c1.z,
      __cgl_mat_arg_0.c2.x, __cgl_mat_arg_0.c2.y, __cgl_mat_arg_0.c2.z)
};
let mut worldPosition
    : Vec4<f32> =
          (modelMatrix * Vec4::<f32>::new (input.position.x, input.position.y,
                                           input.position.z, 1.0));
let worldNormal : Vec3<f32> = normalize((normalMatrix * input.normal));
let worldTangent : Vec3<f32> = normalize((normalMatrix * input.tangent));
let worldBitangent : Vec3<f32> = normalize((normalMatrix * input.bitangent));
let TBN : Mat3<f32> = Mat3::<f32>::new (
              worldTangent.x, worldTangent.y, worldTangent.z, worldBitangent.x,
              worldBitangent.y, worldBitangent.z, worldNormal.x, worldNormal.y,
              worldNormal.z);
let displacement
    : f32 = (fbm(
                 {
                   let __cgl_vec_arg_7 = ((*GLOBALS).scene.time * 0.1);
                   Vec3::<f32>::new ((worldPosition.x + __cgl_vec_arg_7),
                                     (worldPosition.y + __cgl_vec_arg_7),
                                     (worldPosition.z + __cgl_vec_arg_7))
                 },
                 4, 2.0, 0.5) *
             0.1);
if (input.materialIndex > 0) {
  {
    let __cgl_swizzle_0 = Vec3::<f32>::new ((worldNormal.x * displacement),
                                            (worldNormal.y * displacement),
                                            (worldNormal.z * displacement));
    worldPosition.x += __cgl_swizzle_0.x;
    worldPosition.y += __cgl_swizzle_0.y;
    worldPosition.z += __cgl_swizzle_0.z;
  };
}
let viewDir : Vec3<f32> =
                  normalize(((*GLOBALS).cameraPosition -
                             Vec3::<f32>::new (worldPosition.x, worldPosition.y,
                                               worldPosition.z)));
let fresnel : f32 = pow((1.0 - max(0.0, dot(worldNormal, viewDir))), 5.0);
if (input.materialIndex < (*GLOBALS).scene.activeLightCount) {
  output.color = { let __cgl_vec_arg_8 = Vec4::<f32>::new (1.0, 1.0, 1.0, 1.0);
  Vec4::<f32>::new (
      (input.color.x * __cgl_vec_arg_8.x), (input.color.y * __cgl_vec_arg_8.y),
      (input.color.z * __cgl_vec_arg_8.z), (input.color.w * __cgl_vec_arg_8.w))
};
let mut i : i32 = 0;
while (i < 4) {
  if (i >= ((*GLOBALS).frameCount % 5)) {
  }
  let light : Light = (*GLOBALS).scene.lights[i as usize];
  let lightDir
      : Vec3<f32> = normalize((
            light.position - Vec3::<f32>::new (worldPosition.x, worldPosition.y,
                                               worldPosition.z)));
  let lightDistance
      : f32 = length((light.position - Vec3::<f32>::new (worldPosition.x,
                                                         worldPosition.y,
                                                         worldPosition.z)));
  let attenuation : f32 = (1.0 / (1.0 + (lightDistance * lightDistance)));
  let lightIntensity : f32 = (light.intensity * attenuation);
  {
    let __cgl_swizzle_1 = {
      let __cgl_vec_arg_11 = {let __cgl_vec_arg_9 = Vec3::<f32>::new (
                                  (light.color.x * lightIntensity),
                                  (light.color.y * lightIntensity),
                                  (light.color.z * lightIntensity));
    let __cgl_vec_arg_10 = max(0.0, dot(worldNormal, lightDir));
    Vec3::<f32>::new ((__cgl_vec_arg_9.x * __cgl_vec_arg_10),
                      (__cgl_vec_arg_9.y * __cgl_vec_arg_10),
                      (__cgl_vec_arg_9.z * __cgl_vec_arg_10))
  };
  Vec3::<f32>::new ((__cgl_vec_arg_11.x * 0.025), (__cgl_vec_arg_11.y * 0.025),
                    (__cgl_vec_arg_11.z * 0.025))
};
output.color.x += __cgl_swizzle_1.x;
output.color.y += __cgl_swizzle_1.y;
output.color.z += __cgl_swizzle_1.z;
}
;
i += 1;
}
}
else {
  output.color = input.color;
  if ((*GLOBALS).globalRoughness > 0.5) {
    if (fresnel > 0.7) {
      output.color.w *= 0.8;
    } else {
      output.color.w *= 0.9;
    }
  }
}
output.worldPosition =
    Vec3::<f32>::new (worldPosition.x, worldPosition.y, worldPosition.z);
output.worldNormal = worldNormal;
output.worldTangent = worldTangent;
output.worldBitangent = worldBitangent;
output.texCoord0 = input.texCoord0;
output.texCoord1 = input.texCoord1;
output.TBN = TBN;
output.materialIndex = input.materialIndex;
output.clipPosition = (modelViewProjectionMatrix *
                       Vec4::<f32>::new (input.position.x, input.position.y,
                                         input.position.z, 1.0));
return output;
}

// CrossGL resource metadata: name=shadowMap kind=texture set=0 binding=0
// binding_source=automatic
static SHADOW_MAP : std::sync::LazyLock<Texture2D<f32>> =
                        std::sync::LazyLock::new (|| Default::default());
// Fragment Shader
#[cfg_attr(feature = "crossgl_gpu", fragment_shader)]
pub fn fragment_main(input : VertexOutput) -> FragmentOutput {
  let mut output : FragmentOutput = Default::default();
  let material : Material =
                     (*GLOBALS).scene.materials[input.materialIndex as usize];
  let albedoValue : Vec4<f32> = sample(material.albedoMap, input.texCoord0);
  let normalValue : Vec4<f32> = sample(material.normalMap, input.texCoord0);
  let metallicRoughnessValue
      : Vec4<f32> = sample(material.metallicRoughnessMap, input.texCoord0);
  let normal : Vec3<f32> = {
    let __cgl_vec_arg_12 = Vec3::<f32>::new (
        (normalValue.x * 2.0), (normalValue.y * 2.0), (normalValue.z * 2.0));
  Vec3::<f32>::new ((__cgl_vec_arg_12.x - 1.0), (__cgl_vec_arg_12.y - 1.0),
                    (__cgl_vec_arg_12.z - 1.0))
};
let worldNormal : Vec3<f32> = normalize((input.TBN * normal));
let albedo
    : Vec3<f32> =
          (Vec3::<f32>::new (albedoValue.x, albedoValue.y, albedoValue.z) *
           material.albedo);
let metallic : f32 = (metallicRoughnessValue.z * material.metallic);
let roughness : f32 = (metallicRoughnessValue.y * material.roughness);
let ao : f32 = metallicRoughnessValue.x;
let viewDir : Vec3<f32> =
                  normalize(((*GLOBALS).cameraPosition - input.worldPosition));
let F0 : Vec3<f32> =
             lerp(Vec3::<f32>::new (0.04, 0.04, 0.04), albedo, metallic);
let mut Lo : Vec3<f32> = Vec3::<f32>::new (0.0, 0.0, 0.0);
let mut i : i32 = 0;
while (i < (*GLOBALS).scene.activeLightCount) {
  if (i >= 8) {
  }
  let light : Light = (*GLOBALS).scene.lights[i as usize];
  let lightDir : Vec3<f32> = normalize((light.position - input.worldPosition));
  let halfway : Vec3<f32> = normalize((viewDir + lightDir));
  let distance : f32 = length((light.position - input.worldPosition));
  let attenuation : f32 = (1.0 / (distance * distance));
  let radiance : Vec3<f32> = {
    let __cgl_vec_arg_13 = Vec3::<f32>::new ((light.color.x * light.intensity),
                                             (light.color.y * light.intensity),
                                             (light.color.z * light.intensity));
  Vec3::<f32>::new ((__cgl_vec_arg_13.x * attenuation),
                    (__cgl_vec_arg_13.y * attenuation),
                    (__cgl_vec_arg_13.z * attenuation))
};
let NDF : f32 = distributionGGX(worldNormal, halfway, roughness);
let G : f32 = geometrySmith(worldNormal, viewDir, lightDir, roughness);
let F : Vec3<f32> = fresnelSchlick(max(dot(halfway, viewDir), 0.0), F0);
let kS : Vec3<f32> = F;
let mut kD : Vec3<f32> = (Vec3::<f32>::new (1.0, 1.0, 1.0) - kS);
kD = { let __cgl_vec_arg_14 = (1.0 - metallic);
Vec3::<f32>::new ((kD.x * __cgl_vec_arg_14), (kD.y * __cgl_vec_arg_14),
                  (kD.z * __cgl_vec_arg_14))
}
;
let numerator : Vec3<f32> = { let __cgl_vec_arg_15 = (NDF * G);
Vec3::<f32>::new ((__cgl_vec_arg_15 * F.x), (__cgl_vec_arg_15 * F.y),
                  (__cgl_vec_arg_15 * F.z))
}
;
let denominator : f32 = (((4.0 * max(dot(worldNormal, viewDir), 0.0)) *
                          max(dot(worldNormal, lightDir), 0.0)) +
                         EPSILON);
let specular : Vec3<f32> = Vec3::<f32>::new ((numerator.x / denominator),
                                             (numerator.y / denominator),
                                             (numerator.z / denominator));
let NdotL : f32 = max(dot(worldNormal, lightDir), 0.0);
let mut shadow : f32 = 0.0;
if light
  .castShadows {
    let fragPosLightSpace
        : Vec4<f32> =
              (light.viewProjection *
               Vec4::<f32>::new (input.worldPosition.x, input.worldPosition.y,
                                 input.worldPosition.z, 1.0));
    shadow = shadowCalculation(fragPosLightSpace, 0, input);
    let mut s : i32 = 0;
    while (s < 4) {
      if (s >= ((*GLOBALS).frameCount % 3)) {
      }
      shadow += shadowCalculation(
          (fragPosLightSpace +
           Vec4::<f32>::new (
               ((*GLOBALS).noiseValues[(s % 16) as usize] * 0.001), 0.0, 0.0,
               0.0)),
          (s + 1), input);
      s += 1;
    }
    shadow /= 5.0;
  }
Lo = (Lo + {
  let __cgl_vec_arg_19 = ({
    let __cgl_vec_arg_16 = (1.0 - shadow);
    let __cgl_vec_arg_18 = ({
      let __cgl_vec_arg_17 = (kD * albedo);
      Vec3::<f32>::new ((__cgl_vec_arg_17.x / PI), (__cgl_vec_arg_17.y / PI),
                        (__cgl_vec_arg_17.z / PI))
    } + specular);
    Vec3::<f32>::new ((__cgl_vec_arg_16 * __cgl_vec_arg_18.x),
                      (__cgl_vec_arg_16 * __cgl_vec_arg_18.y),
                      (__cgl_vec_arg_16 * __cgl_vec_arg_18.z))
  } * radiance);
  Vec3::<f32>::new ((__cgl_vec_arg_19.x * NdotL), (__cgl_vec_arg_19.y * NdotL),
                    (__cgl_vec_arg_19.z * NdotL))
});
i += 1;
}
let ambient : Vec3<f32> = {
  let __cgl_vec_arg_20 = ((*GLOBALS).scene.ambientLight * albedo);
Vec3::<f32>::new ((__cgl_vec_arg_20.x * ao), (__cgl_vec_arg_20.y * ao),
                  (__cgl_vec_arg_20.z * ao))
}
;
let mut color : Vec3<f32> = (ambient + Lo);
color = { let __cgl_vec_arg_21 = (color + Vec3::<f32>::new (1.0, 1.0, 1.0));
Vec3::<f32>::new ((color.x / __cgl_vec_arg_21.x),
                  (color.y / __cgl_vec_arg_21.y),
                  (color.z / __cgl_vec_arg_21.z))
}
;
color = pow(color, {
  let __cgl_vec_arg_22 = (1.0 / 2.2);
  Vec3::<f32>::new (__cgl_vec_arg_22, __cgl_vec_arg_22, __cgl_vec_arg_22)
});
output.color = Vec4::<f32>::new (color.x, color.y, color.z,
                                 (material.opacity * albedoValue.w));
output.normalBuffer = {
  let __cgl_vec_arg_24 = {
      let __cgl_vec_arg_23 = Vec3::<f32>::new (
          (worldNormal.x * 0.5), (worldNormal.y * 0.5), (worldNormal.z * 0.5));
Vec3::<f32>::new ((__cgl_vec_arg_23.x + 0.5), (__cgl_vec_arg_23.y + 0.5),
                  (__cgl_vec_arg_23.z + 0.5))
}
;
Vec4::<f32>::new (__cgl_vec_arg_24.x, __cgl_vec_arg_24.y, __cgl_vec_arg_24.z,
                  1.0)
}
;
output.positionBuffer = Vec4::<f32>::new (
    input.worldPosition.x, input.worldPosition.y, input.worldPosition.z, 1.0);
output.depth = (input.clipPosition.z / input.clipPosition.w);
return output;
}

pub fn shadowCalculation(fragPosLightSpace : Vec4<f32>, iteration : i32,
                         input : VertexOutput) -> f32 {
  if (iteration > 3) {
  }
  let mut projCoords
      : Vec3<f32> =
            Vec3::<f32>::new ((fragPosLightSpace.x / fragPosLightSpace.w),
                              (fragPosLightSpace.y / fragPosLightSpace.w),
                              (fragPosLightSpace.z / fragPosLightSpace.w));
  projCoords = {
    let __cgl_vec_arg_25 = Vec3::<f32>::new (
        (projCoords.x * 0.5), (projCoords.y * 0.5), (projCoords.z * 0.5));
  Vec3::<f32>::new ((__cgl_vec_arg_25.x + 0.5), (__cgl_vec_arg_25.y + 0.5),
                    (__cgl_vec_arg_25.z + 0.5))
};
let closestDepth
    : f32 =
          sample(*SHADOW_MAP, Vec2::<f32>::new (projCoords.x, projCoords.y)).x;
let currentDepth : f32 = projCoords.z;
let bias : f32 = max((0.05 * (1.0 - dot(input.worldNormal,
                                        normalize(((*GLOBALS).cameraPosition -
                                                   input.worldPosition))))),
                     0.005);
let mut shadow
    : f32 = (if ((currentDepth - bias) > closestDepth){1.0} else {0.0});
let pcfDepth : f32 = 0.0;
let texelSize : Vec2<f32> = {
  let __cgl_vec_arg_26 =
      Vec2::<f32>::new ((*GLOBALS).screenSize.x, (*GLOBALS).screenSize.y);
Vec2::<f32>::new ((1.0 / __cgl_vec_arg_26.x), (1.0 / __cgl_vec_arg_26.y))
}
;
let offset
    : f32 = ((*GLOBALS).noiseValues[((iteration * 4) % 16) as usize] * 0.001);
let mut x : i32 = (-1);
while (x <= 1) {
  let mut y : i32 = (-1);
  while (y <= 1) {
            let pcfDepth: f32 = sample(*SHADOW_MAP, {
      let __cgl_vec_arg_29 = {
        let __cgl_vec_arg_28 = {let __cgl_vec_arg_27 =
                                    Vec2::<f32>::new ((x as f32), (y as f32));
      Vec2::<f32>::new ((__cgl_vec_arg_27.x * texelSize.x),
                        (__cgl_vec_arg_27.y * texelSize.y)) }; Vec2::<f32>::new((projCoords.x + __cgl_vec_arg_28.x), (projCoords.y + __cgl_vec_arg_28.y))
  };
  let __cgl_vec_arg_30 = Vec2::<f32>::new (offset, offset);
  Vec2::<f32>::new ((__cgl_vec_arg_29.x + __cgl_vec_arg_30.x),
                    (__cgl_vec_arg_29.y + __cgl_vec_arg_30.y))
}).x;
shadow += (if ((currentDepth - bias) > pcfDepth){1.0} else {0.0});
y += 1;
}
x += 1;
}
shadow /= 9.0;
if (projCoords.z > 1.0) {
  shadow = 0.0;
}
return shadow;
}

// CrossGL resource metadata: name=outputImage kind=image set=0 binding=0
// binding_source=automatic
static OUTPUT_IMAGE : std::sync::LazyLock<Image2D<Vec4<f32>>> =
                          std::sync::LazyLock::new (|| Default::default());
// Compute Shader
#[cfg_attr(feature = "crossgl_gpu", compute_shader)]
pub fn compute_main() -> () {
  let texCoord : Vec2<i32> =
                     Vec2::<i32>::new ((global_invocation_id().x as i32),
                                       (global_invocation_id().y as i32));
  let screenSize : Vec2<f32> = (*GLOBALS).screenSize;
  if ((texCoord.x >= (screenSize.x as i32)) ||
      (texCoord.y >= (screenSize.y as i32))) {
    return;
  }
  let uv : Vec2<f32> = {
    let __cgl_vec_arg_31 =
        Vec2::<f32>::new ((texCoord.x as f32), (texCoord.y as f32));
  Vec2::<f32>::new ((__cgl_vec_arg_31.x / screenSize.x),
                    (__cgl_vec_arg_31.y / screenSize.y))
};
let mut color : Vec4<f32> = Vec4::<f32>::new (0.0, 0.0, 0.0, 0.0);
let mut totalWeight : f32 = 0.0;
let mut direction : Vec2<f32> = {
  let __cgl_vec_arg_32 = Vec2::<f32>::new (0.5, 0.5);
Vec2::<f32>::new ((__cgl_vec_arg_32.x - uv.x), (__cgl_vec_arg_32.y - uv.y))
}
;
let len : f32 = { let __cgl_vec_arg_33 : Vec2<f32> = direction;
(((__cgl_vec_arg_33.x * __cgl_vec_arg_33.x) +
  (__cgl_vec_arg_33.y * __cgl_vec_arg_33.y)))
    .sqrt()
}
;
direction = { let __cgl_vec_arg_34 : Vec2<f32> = direction;
let __cgl_vec_arg_35 : f32 = (((__cgl_vec_arg_34.x * __cgl_vec_arg_34.x) +
                               (__cgl_vec_arg_34.y * __cgl_vec_arg_34.y)))
                                 .sqrt();
Vec2::<f32>::new ((__cgl_vec_arg_34.x / __cgl_vec_arg_35),
                  (__cgl_vec_arg_34.y / __cgl_vec_arg_35))
}
;
let mut i : i32 = 0;
while (i < 32) {
  if (i >= MAX_ITERATIONS) {
  }
  let t : f32 = ((i as f32) / 32.0);
  let pos : Vec2<f32> = {
    let __cgl_vec_arg_38 = {
        let __cgl_vec_arg_37 = {let __cgl_vec_arg_36 = Vec2::<f32>::new (
                                    (direction.x * t), (direction.y * t));
  Vec2::<f32>::new ((__cgl_vec_arg_36.x * len), (__cgl_vec_arg_36.y * len))
};
Vec2::<f32>::new ((__cgl_vec_arg_37.x * 0.1), (__cgl_vec_arg_37.y * 0.1))
}
;
Vec2::<f32>::new ((uv.x + __cgl_vec_arg_38.x), (uv.y + __cgl_vec_arg_38.y))
}
;
let noise : f32 = fbm(
                {
                  let __cgl_vec_arg_39 =
                      Vec2::<f32>::new ((pos.x * 10.0), (pos.y * 10.0));
                  Vec3::<f32>::new (__cgl_vec_arg_39.x, __cgl_vec_arg_39.y,
                                    ((*GLOBALS).scene.time * 0.05))
                },
                4, 2.0, 0.5);
let mut weight : f32 = (1.0 - t);
weight = (weight * weight);
let noiseColor
    : Vec3<f32> = Vec3::<f32>::new (
          (0.5 + (0.5 * sin((((noise * 5.0) + (*GLOBALS).scene.time) + 0.0)))),
          (0.5 + (0.5 * sin((((noise * 5.0) + (*GLOBALS).scene.time) + 2.0)))),
          (0.5 + (0.5 * sin((((noise * 5.0) + (*GLOBALS).scene.time) + 4.0)))));
{
  let __cgl_swizzle_2 =
      Vec3::<f32>::new ((noiseColor.x * weight), (noiseColor.y * weight),
                        (noiseColor.z * weight));
  color.x += __cgl_swizzle_2.x;
  color.y += __cgl_swizzle_2.y;
  color.z += __cgl_swizzle_2.z;
};
totalWeight += weight;
direction = {
  let __cgl_mat_arg_1 = Mat2::<f32>::new (cos((t * 3.0)), (-sin((t * 3.0))),
                                          sin((t * 3.0)), cos((t * 3.0)));
Vec2::<f32>::new (((__cgl_mat_arg_1.c0.x * direction.x) +
                   (__cgl_mat_arg_1.c1.x * direction.y)),
                  ((__cgl_mat_arg_1.c0.y * direction.x) +
                   (__cgl_mat_arg_1.c1.y * direction.y)))
}
;
i += 1;
}
{
  let __cgl_swizzle_3 = totalWeight;
  color.x /= __cgl_swizzle_3;
  color.y /= __cgl_swizzle_3;
  color.z /= __cgl_swizzle_3;
};
color.w = 1.0;
let vignette
    : f32 = (1.0 - smoothstep(0.5, 1.0, ({
                                let __cgl_vec_arg_40
                                    : Vec2<f32> = Vec2::<f32>::new (
                                          (uv.x - 0.5), (uv.y - 0.5));
                                (((__cgl_vec_arg_40.x * __cgl_vec_arg_40.x) +
                                  (__cgl_vec_arg_40.y * __cgl_vec_arg_40.y)))
                                    .sqrt()
                              } * 1.5)));
{
  let __cgl_swizzle_4 = vignette;
  color.x *= __cgl_swizzle_4;
  color.y *= __cgl_swizzle_4;
  color.z *= __cgl_swizzle_4;
};
image_store(*OUTPUT_IMAGE, texCoord, color);
}
