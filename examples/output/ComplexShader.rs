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
        pub viewProjection : Mat4<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Scene {
  pub materials : [Material; 4],
                  pub lights : [Light; 8],
                               pub ambientLight : Vec3<f32>,
                                                  pub time : f32,
                                                             pub elapsedTime
      : f32,
        pub activeLightCount : i32,
                               pub viewMatrix : Mat4<f32>,
                                                pub projectionMatrix
      : Mat4<f32>,
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
                                                    pub TBN : Mat3<f32>,
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
      : Vec<f32>,
}

static PI : f32 = Default::default();
static EPSILON : f32 = Default::default();
static MAX_ITERATIONS : i32 = Default::default();
static a2 : f32 = Default::default();
static num : f32 = Default::default();
static denom : f32 = Default::default();
pub fn geometrySchlickGGX(NdotV : f32, roughness : f32) -> f32 {
  (r = (roughness + 1.0));
  (k = ((r * r) / 8.0));
  (num = NdotV);
  (denom = ((NdotV * (1.0 - k)) + k));
  return ((num / max(denom, EPSILON)));
}

pub fn geometrySmith(N : Vec3<f32>, V : Vec3<f32>, L : Vec3<f32>,
                     roughness : f32) -> f32 {
  (NdotV = max(dot(N, V), 0.0));
  (ggx2 = geometrySchlickGGX(NdotV, roughness));
  return ((ggx1 * ggx2));
}

pub fn fresnelSchlick(cosTheta : f32, F0 : Vec3<f32>) -> Vec3<f32> {
  return ((F0 + ((1.0 - F0) * pow(max((1.0 - cosTheta), 0.0), 5.0))));
}

pub fn noise3D(p : Vec3<f32>) -> f32 {
  (i = floor(p));
  (u = (((f * f) * f) * ((f * ((f * 6.0) - 15.0)) + 10.0)));
  (n000 =
       fract((sin(dot(i, Vec3<f32>::new (13.534, 43.5234, 243.32))) * 4453.0)));
  (n010 = fract((sin(dot((i + Vec3<f32>::new (0.0, 1.0, 0.0)),
                         Vec3<f32>::new (13.534, 43.5234, 243.32))) *
                 4453.0)));
  (n100 = fract((sin(dot((i + Vec3<f32>::new (1.0, 0.0, 0.0)),
                         Vec3<f32>::new (13.534, 43.5234, 243.32))) *
                 4453.0)));
  (n110 = fract((sin(dot((i + Vec3<f32>::new (1.0, 1.0, 0.0)),
                         Vec3<f32>::new (13.534, 43.5234, 243.32))) *
                 4453.0)));
  (n00 = lerp(n000, n001, u.z));
  (n10 = lerp(n100, n101, u.z));
  (n0 = lerp(n00, n01, u.y));
  return (lerp(n0, n1, u.x));
}

pub fn fbm(p : Vec3<f32>, octaves : i32, lacunarity : f32, gain : f32) -> f32 {
  (sum = 0.0);
  (amplitude = 1.0);
  (frequency = 1.0);
  (i = 0);
  ;
  while (i < octaves) {
    if (i >= MAX_ITERATIONS) {
      let break;
    }
    (sum += (amplitude * noise3D((p * frequency))));
    (frequency *= lacunarity);
    i++;
  }
  return (sum);
}

pub fn samplePlanarProjection(tex : Texture2D<f32>, worldPos : Vec3<f32>,
                              normal : Vec3<f32>) -> Vec4<f32> {
  (absNormal = abs(normal));
  (useY = ((!useX) && (absNormal.y >= absNormal.z)));
  let uv : Vec2<f32>;
  if useX {
    (uv = ((worldPos.zy * 0.5) + 0.5));
    if (normal.x < 0.0) {
      (MemberAccessNode(object = uv, member = x) = (1.0 - uv.x));
    }
  } else if useY {
    (uv = ((worldPos.xz * 0.5) + 0.5));
    if (normal.y < 0.0) {
      (MemberAccessNode(object = uv, member = y) = (1.0 - uv.y));
    }
  } else {
    (uv = ((worldPos.xy * 0.5) + 0.5));
    if (normal.z < 0.0) {
      (MemberAccessNode(object = uv, member = x) = (1.0 - uv.x));
    }
  }
  return (sample(tex, uv));
}
