# Generated Mojo Shader Code
from math import *
from simd import *
from gpu import *

@value
struct Material:
    var albedo: SIMD[DType.float32, 3]
    var roughness: Float32
    var metallic: Float32
    var emissive: SIMD[DType.float32, 3]
    var opacity: Float32
    var hasNormalMap: Bool
    var albedoMap: Texture2D
    var normalMap: Texture2D
    var metallicRoughnessMap: Texture2D

@value
struct Light:
    var position: SIMD[DType.float32, 3]
    var color: SIMD[DType.float32, 3]
    var intensity: Float32
    var radius: Float32
    var castShadows: Bool
    var viewProjection: Matrix[DType.float32, 4, 4]

@value
struct Scene:
    var materials: StaticTuple[Material, 4]
    var lights: StaticTuple[Light, 8]
    var ambientLight: SIMD[DType.float32, 3]
    var time: Float32
    var elapsedTime: Float32
    var activeLightCount: Int32
    var viewMatrix: Matrix[DType.float32, 4, 4]
    var projectionMatrix: Matrix[DType.float32, 4, 4]

@value
struct VertexInput:
    var position: SIMD[DType.float32, 3]
    var normal: SIMD[DType.float32, 3]
    var tangent: SIMD[DType.float32, 3]
    var bitangent: SIMD[DType.float32, 3]
    var texCoord0: SIMD[DType.float32, 2]
    var texCoord1: SIMD[DType.float32, 2]
    var color: SIMD[DType.float32, 4]
    var materialIndex: Int32

@value
struct VertexOutput:
    var worldPosition: SIMD[DType.float32, 3]
    var worldNormal: SIMD[DType.float32, 3]
    var worldTangent: SIMD[DType.float32, 3]
    var worldBitangent: SIMD[DType.float32, 3]
    var texCoord0: SIMD[DType.float32, 2]
    var texCoord1: SIMD[DType.float32, 2]
    var color: SIMD[DType.float32, 4]
    var TBN: Matrix[DType.float32, 3, 3]
    var materialIndex: Int32
    var clipPosition: SIMD[DType.float32, 4]

@value
struct FragmentOutput:
    var color: SIMD[DType.float32, 4]
    var normalBuffer: SIMD[DType.float32, 4]
    var positionBuffer: SIMD[DType.float32, 4]
    var depth: Float32

@value
struct GlobalUniforms:
    var scene: Scene
    var cameraPosition: SIMD[DType.float32, 3]
    var globalRoughness: Float32
    var screenSize: SIMD[DType.float32, 2]
    var nearPlane: Float32
    var farPlane: Float32
    var frameCount: Int32
    var noiseValues: DynamicVector[Float32]

var PI: Float32
var EPSILON: Float32
var MAX_ITERATIONS: Int32
var a2: Float32
var num: Float32
var denom: Float32
fn geometrySchlickGGX(NdotV: Float32, roughness: Float32) -> Float32:
    (r = (roughness + 1.0))
    (k = ((r * r) / 8.0))
    (num = NdotV)
    (denom = ((NdotV * (1.0 - k)) + k))
    return (num / max(denom, EPSILON))

fn geometrySmith(N: SIMD[DType.float32, 3], V: SIMD[DType.float32, 3], L: SIMD[DType.float32, 3], roughness: Float32) -> Float32:
    (NdotV = max(dot_product(N, V), 0.0))
    (ggx2 = geometrySchlickGGX(NdotV, roughness))
    return (ggx1 * ggx2)

fn fresnelSchlick(cosTheta: Float32, F0: SIMD[DType.float32, 3]) -> SIMD[DType.float32, 3]:
    return (F0 + ((1.0 - F0) * power(max((1.0 - cosTheta), 0.0), 5.0)))

fn noise3D(p: SIMD[DType.float32, 3]) -> Float32:
    (i = floor(p))
    (u = (((f * f) * f) * ((f * ((f * 6.0) - 15.0)) + 10.0)))
    (n000 = fract((sin(dot_product(i, SIMD[DType.float32, 3](13.534, 43.5234, 243.32))) * 4453.0)))
    (n010 = fract((sin(dot_product((i + SIMD[DType.float32, 3](0.0, 1.0, 0.0)), SIMD[DType.float32, 3](13.534, 43.5234, 243.32))) * 4453.0)))
    (n100 = fract((sin(dot_product((i + SIMD[DType.float32, 3](1.0, 0.0, 0.0)), SIMD[DType.float32, 3](13.534, 43.5234, 243.32))) * 4453.0)))
    (n110 = fract((sin(dot_product((i + SIMD[DType.float32, 3](1.0, 1.0, 0.0)), SIMD[DType.float32, 3](13.534, 43.5234, 243.32))) * 4453.0)))
    (n00 = lerp(n000, n001, u.z))
    (n10 = lerp(n100, n101, u.z))
    (n0 = lerp(n00, n01, u.y))
    return lerp(n0, n1, u.x)

fn fbm(p: SIMD[DType.float32, 3], octaves: Int32, lacunarity: Float32, gain: Float32) -> Float32:
    (sum = 0.0)
    (amplitude = 1.0)
    (frequency = 1.0)
    (i = 0)
    while (i < octaves):
        if (i >= MAX_ITERATIONS):
            var break
        (sum += (amplitude * noise3D((p * frequency))))
        (frequency *= lacunarity)
        i++
    return sum

fn samplePlanarProjection(tex: Texture2D, worldPos: SIMD[DType.float32, 3], normal: SIMD[DType.float32, 3]) -> SIMD[DType.float32, 4]:
    (absNormal = abs(normal))
    (useY = ((!useX) && (absNormal.y >= absNormal.z)))
    var uv: SIMD[DType.float32, 2]
    if useX:
        (uv = ((worldPos.zy * 0.5) + 0.5))
        if (normal.x < 0.0):
            (MemberAccessNode(object=uv, member=x) = (1.0 - uv.x))
    elif useY:
        (uv = ((worldPos.xz * 0.5) + 0.5))
        if (normal.y < 0.0):
            (MemberAccessNode(object=uv, member=y) = (1.0 - uv.y))
    else:
        (uv = ((worldPos.xy * 0.5) + 0.5))
        if (normal.z < 0.0):
            (MemberAccessNode(object=uv, member=x) = (1.0 - uv.x))
    return sample(tex, uv)

