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
    var viewProjection: mat4x4

@value
struct Scene:
    var materials: MaterialLiteralNode(value=4, literal_type=PrimitiveType(name=int, size_bits=None))
    var lights: LightLiteralNode(value=8, literal_type=PrimitiveType(name=int, size_bits=None))
    var ambientLight: SIMD[DType.float32, 3]
    var time: Float32
    var elapsedTime: Float32
    var activeLightCount: Int32
    var viewMatrix: mat4x4
    var projectionMatrix: mat4x4

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
    var TBN: mat3x3
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
    var noiseValues: vecNone

var PI: Float32
var EPSILON: Float32
var MAX_ITERATIONS: Int32
var UP_VECTOR: VectorType(element_type=PrimitiveType(name=float, size_bits=None), size=3)
fn distributionGGX(N: SIMD[DType.float32, 3], H: SIMD[DType.float32, 3], roughness: Float32) -> Float32:
    var a: Float32 = (roughness * roughness)
    var a2: Float32 = (a * a)
    var NdotH: Float32 = max(dot_product(N, H), 0.0)
    var NdotH2: Float32 = (NdotH * NdotH)
    var num: Float32 = a2
    var denom: Float32 = ((NdotH2 * (a2 - 1.0)) + 1.0)
    denom = ((PI * denom) * denom)
    return (num / max(denom, EPSILON))

fn geometrySchlickGGX(NdotV: Float32, roughness: Float32) -> Float32:
    var r: Float32 = (roughness + 1.0)
    var k: Float32 = ((r * r) / 8.0)
    var num: Float32 = NdotV
    var denom: Float32 = ((NdotV * (1.0 - k)) + k)
    return (num / max(denom, EPSILON))

fn geometrySmith(N: SIMD[DType.float32, 3], V: SIMD[DType.float32, 3], L: SIMD[DType.float32, 3], roughness: Float32) -> Float32:
    var NdotV: Float32 = max(dot_product(N, V), 0.0)
    var NdotL: Float32 = max(dot_product(N, L), 0.0)
    var ggx2: Float32 = geometrySchlickGGX(NdotV, roughness)
    var ggx1: Float32 = geometrySchlickGGX(NdotL, roughness)
    return (ggx1 * ggx2)

fn fresnelSchlick(cosTheta: Float32, F0: SIMD[DType.float32, 3]) -> SIMD[DType.float32, 3]:
    return (F0 + ((1.0 - F0) * power(max((1.0 - cosTheta), 0.0), 5.0)))

fn noise3D(p: SIMD[DType.float32, 3]) -> Float32:
    var i: SIMD[DType.float32, 3] = floor(p)
    var f: SIMD[DType.float32, 3] = fract(p)
    var u: SIMD[DType.float32, 3] = (((f * f) * f) * ((f * ((f * 6.0) - 15.0)) + 10.0))
    var n000: Float32 = fract((sin(dot_product(i, SIMD[DType.float32, 3](13.534, 43.5234, 243.32))) * 4453.0))
    var n001: Float32 = fract((sin(dot_product((i + SIMD[DType.float32, 3](0.0, 0.0, 1.0)), SIMD[DType.float32, 3](13.534, 43.5234, 243.32))) * 4453.0))
    var n010: Float32 = fract((sin(dot_product((i + SIMD[DType.float32, 3](0.0, 1.0, 0.0)), SIMD[DType.float32, 3](13.534, 43.5234, 243.32))) * 4453.0))
    var n011: Float32 = fract((sin(dot_product((i + SIMD[DType.float32, 3](0.0, 1.0, 1.0)), SIMD[DType.float32, 3](13.534, 43.5234, 243.32))) * 4453.0))
    var n100: Float32 = fract((sin(dot_product((i + SIMD[DType.float32, 3](1.0, 0.0, 0.0)), SIMD[DType.float32, 3](13.534, 43.5234, 243.32))) * 4453.0))
    var n101: Float32 = fract((sin(dot_product((i + SIMD[DType.float32, 3](1.0, 0.0, 1.0)), SIMD[DType.float32, 3](13.534, 43.5234, 243.32))) * 4453.0))
    var n110: Float32 = fract((sin(dot_product((i + SIMD[DType.float32, 3](1.0, 1.0, 0.0)), SIMD[DType.float32, 3](13.534, 43.5234, 243.32))) * 4453.0))
    var n111: Float32 = fract((sin(dot_product((i + SIMD[DType.float32, 3](1.0, 1.0, 1.0)), SIMD[DType.float32, 3](13.534, 43.5234, 243.32))) * 4453.0))
    var n00: Float32 = lerp(n000, n001, u.z)
    var n01: Float32 = lerp(n010, n011, u.z)
    var n10: Float32 = lerp(n100, n101, u.z)
    var n11: Float32 = lerp(n110, n111, u.z)
    var n0: Float32 = lerp(n00, n01, u.y)
    var n1: Float32 = lerp(n10, n11, u.y)
    return lerp(n0, n1, u.x)

fn fbm(p: SIMD[DType.float32, 3], octaves: Int32, lacunarity: Float32, gain: Float32) -> Float32:
    var sum: Float32 = 0.0
    var amplitude: Float32 = 1.0
    var frequency: Float32 = 1.0
    var i: Int32 = 0
    while (i < octaves):
        if (i >= MAX_ITERATIONS):
        sum += (amplitude * noise3D((p * frequency)))
        amplitude *= gain
        frequency *= lacunarity
        (++i)
    return sum

fn samplePlanarProjection(tex: Texture2D, worldPos: SIMD[DType.float32, 3], normal: SIMD[DType.float32, 3]) -> SIMD[DType.float32, 4]:
    var absNormal: SIMD[DType.float32, 3] = abs(normal)
    var useX: Bool = ((absNormal.x >= absNormal.y) && (absNormal.x >= absNormal.z))
    var useY: Bool = ((!useX) && (absNormal.y >= absNormal.z))
    var uv: SIMD[DType.float32, 2]
    if useX:
        uv = ((worldPos.zy * 0.5) + 0.5)
        if (normal.x < 0.0):
    elif useY:
        uv = ((worldPos.xz * 0.5) + 0.5)
        if (normal.y < 0.0):
    else:
        uv = ((worldPos.xy * 0.5) + 0.5)
        if (normal.z < 0.0):
    return sample(tex, uv)

# Vertex Shader
@vertex_shader
fn main(input: VertexInput) -> VertexOutput:
    var output: VertexOutput
    var modelMatrix: mat4x4 = mat4(1.0)
    var viewMatrix: mat4x4 = globals.scene.viewMatrix
    var projectionMatrix: mat4x4 = globals.scene.projectionMatrix
    var modelViewMatrix: mat4x4 = (viewMatrix * modelMatrix)
    var modelViewProjectionMatrix: mat4x4 = (projectionMatrix * modelViewMatrix)
    var normalMatrix: mat3x3 = mat3(transpose(inverse(modelMatrix)))
    var worldPosition: SIMD[DType.float32, 4] = (modelMatrix * SIMD[DType.float32, 4](input.position, 1.0))
    var worldNormal: SIMD[DType.float32, 3] = normalize((normalMatrix * input.normal))
    var worldTangent: SIMD[DType.float32, 3] = normalize((normalMatrix * input.tangent))
    var worldBitangent: SIMD[DType.float32, 3] = normalize((normalMatrix * input.bitangent))
    var TBN: mat3x3 = mat3(worldTangent, worldBitangent, worldNormal)
    var displacement: Float32 = (fbm((worldPosition.xyz + (globals.scene.time * 0.1)), 4, 2.0, 0.5) * 0.1)
    if (input.materialIndex > 0):
        worldPosition.xyz += (worldNormal * displacement)
    var viewDir: SIMD[DType.float32, 3] = normalize((globals.cameraPosition - worldPosition.xyz))
    var fresnel: Float32 = power((1.0 - max(0.0, dot_product(worldNormal, viewDir))), 5.0)
    if (input.materialIndex < globals.scene.activeLightCount):
        output.color = (input.color * SIMD[DType.float32, 4](1.0, 1.0, 1.0, 1.0))
        var i: Int32 = 0
        while (i < 4):
            if (i >= (globals.frameCount % 5)):
            var light: Light = globals.scene.lights[i]
            var lightDir: SIMD[DType.float32, 3] = normalize((light.position - worldPosition.xyz))
            var lightDistance: Float32 = magnitude((light.position - worldPosition.xyz))
            var attenuation: Float32 = (1.0 / (1.0 + (lightDistance * lightDistance)))
            var lightIntensity: Float32 = (light.intensity * attenuation)
            output.color.rgb += (((light.color * lightIntensity) * max(0.0, dot_product(worldNormal, lightDir))) * 0.025)
            (++i)
    else:
        output.color = input.color
        if (globals.globalRoughness > 0.5):
            if (fresnel > 0.7):
                output.color.a *= 0.8
            else:
                output.color.a *= 0.9
    output.worldPosition = worldPosition.xyz
    output.worldNormal = worldNormal
    output.worldTangent = worldTangent
    output.worldBitangent = worldBitangent
    output.texCoord0 = input.texCoord0
    output.texCoord1 = input.texCoord1
    output.TBN = TBN
    output.materialIndex = input.materialIndex
    output.clipPosition = (modelViewProjectionMatrix * SIMD[DType.float32, 4](input.position, 1.0))
    return output

# Fragment Shader
@fragment_shader
fn main(input: VertexOutput) -> FragmentOutput:
    var output: FragmentOutput
    var material: Material = globals.scene.materials[input.materialIndex]
    var albedoValue: SIMD[DType.float32, 4] = sample(material.albedoMap, input.texCoord0)
    var normalValue: SIMD[DType.float32, 4] = sample(material.normalMap, input.texCoord0)
    var metallicRoughnessValue: SIMD[DType.float32, 4] = sample(material.metallicRoughnessMap, input.texCoord0)
    var normal: SIMD[DType.float32, 3] = ((normalValue.xyz * 2.0) - 1.0)
    var worldNormal: SIMD[DType.float32, 3] = normalize((input.TBN * normal))
    var albedo: SIMD[DType.float32, 3] = (albedoValue.rgb * material.albedo)
    var metallic: Float32 = (metallicRoughnessValue.b * material.metallic)
    var roughness: Float32 = (metallicRoughnessValue.g * material.roughness)
    var ao: Float32 = metallicRoughnessValue.r
    var viewDir: SIMD[DType.float32, 3] = normalize((globals.cameraPosition - input.worldPosition))
    var F0: SIMD[DType.float32, 3] = lerp(SIMD[DType.float32, 3](0.04), albedo, metallic)
    var Lo: SIMD[DType.float32, 3] = SIMD[DType.float32, 3](0.0)
    var i: Int32 = 0
    while (i < globals.scene.activeLightCount):
        if (i >= 8):
        var light: Light = globals.scene.lights[i]
        var lightDir: SIMD[DType.float32, 3] = normalize((light.position - input.worldPosition))
        var halfway: SIMD[DType.float32, 3] = normalize((viewDir + lightDir))
        var distance: Float32 = magnitude((light.position - input.worldPosition))
        var attenuation: Float32 = (1.0 / (distance * distance))
        var radiance: SIMD[DType.float32, 3] = ((light.color * light.intensity) * attenuation)
        var NDF: Float32 = distributionGGX(worldNormal, halfway, roughness)
        var G: Float32 = geometrySmith(worldNormal, viewDir, lightDir, roughness)
        var F: SIMD[DType.float32, 3] = fresnelSchlick(max(dot_product(halfway, viewDir), 0.0), F0)
        var kS: SIMD[DType.float32, 3] = F
        var kD: SIMD[DType.float32, 3] = (SIMD[DType.float32, 3](1.0) - kS)
        kD *= (1.0 - metallic)
        var numerator: SIMD[DType.float32, 3] = ((NDF * G) * F)
        var denominator: Float32 = (((4.0 * max(dot_product(worldNormal, viewDir), 0.0)) * max(dot_product(worldNormal, lightDir), 0.0)) + EPSILON)
        var specular: SIMD[DType.float32, 3] = (numerator / denominator)
        var NdotL: Float32 = max(dot_product(worldNormal, lightDir), 0.0)
        var shadow: Float32 = 0.0
        if light.castShadows:
            var fragPosLightSpace: SIMD[DType.float32, 4] = (light.viewProjection * SIMD[DType.float32, 4](input.worldPosition, 1.0))
            shadow = shadowCalculation(fragPosLightSpace, 0)
            var s: Int32 = 0
            while (s < 4):
                if (s >= (globals.frameCount % 3)):
                shadow += shadowCalculation((fragPosLightSpace + SIMD[DType.float32, 4]((globals.noiseValues[(s % 16)] * 0.001), 0.0, 0.0, 0.0)), (s + 1))
                (++s)
            shadow /= 5.0
        Lo += ((((1.0 - shadow) * (((kD * albedo) / PI) + specular)) * radiance) * NdotL)
        (++i)
    var ambient: SIMD[DType.float32, 3] = ((globals.scene.ambientLight * albedo) * ao)
    var color: SIMD[DType.float32, 3] = (ambient + Lo)
    color = (color / (color + SIMD[DType.float32, 3](1.0)))
    color = power(color, SIMD[DType.float32, 3]((1.0 / 2.2)))
    output.color = SIMD[DType.float32, 4](color, (material.opacity * albedoValue.a))
    output.normalBuffer = SIMD[DType.float32, 4](((worldNormal * 0.5) + 0.5), 1.0)
    output.positionBuffer = SIMD[DType.float32, 4](input.worldPosition, 1.0)
    output.depth = (input.clipPosition.z / input.clipPosition.w)
    return output

fn shadowCalculation(fragPosLightSpace: SIMD[DType.float32, 4], iteration: Int32) -> Float32:
    pass

fn shadowCalculation(fragPosLightSpace: SIMD[DType.float32, 4], iteration: Int32) -> Float32:
    if (iteration > 3):
    var projCoords: SIMD[DType.float32, 3] = (fragPosLightSpace.xyz / fragPosLightSpace.w)
    projCoords = ((projCoords * 0.5) + 0.5)
    var closestDepth: Float32 = sample(shadowMap, projCoords.xy).r
    var currentDepth: Float32 = projCoords.z
    var bias: Float32 = max((0.05 * (1.0 - dot_product(input.worldNormal, normalize((globals.cameraPosition - input.worldPosition))))), 0.005)
    var shadow: Float32 = (1.0 if ((currentDepth - bias) > closestDepth) else 0.0)
    var pcfDepth: Float32 = 0.0
    var texelSize: SIMD[DType.float32, 2] = (1.0 / SIMD[DType.float32, 2](globals.screenSize))
    var offset: Float32 = (globals.noiseValues[((iteration * 4) % 16)] * 0.001)
    var x: Int32 = (-1)
    while (x <= 1):
        var y: Int32 = (-1)
        while (y <= 1):
            var pcfDepth: Float32 = sample(shadowMap, ((projCoords.xy + (SIMD[DType.float32, 2](x, y) * texelSize)) + SIMD[DType.float32, 2](offset))).r
            shadow += (1.0 if ((currentDepth - bias) > pcfDepth) else 0.0)
            (++y)
        (++x)
    shadow /= 9.0
    if (projCoords.z > 1.0):
        shadow = 0.0
    return shadow

# Compute Shader
@compute_shader
fn main() -> None:
    var texCoord: SIMD[DType.int32, 2] = SIMD[DType.int32, 2](gl_GlobalInvocationID.xy)
    var screenSize: SIMD[DType.float32, 2] = globals.screenSize
    if ((texCoord.x >= int(screenSize.x)) || (texCoord.y >= int(screenSize.y))):
        return None
    var uv: SIMD[DType.float32, 2] = (SIMD[DType.float32, 2](texCoord) / screenSize)
    var color: SIMD[DType.float32, 4] = SIMD[DType.float32, 4](0.0)
    var totalWeight: Float32 = 0.0
    var direction: SIMD[DType.float32, 2] = (SIMD[DType.float32, 2](0.5) - uv)
    var len: Float32 = magnitude(direction)
    direction = normalize(direction)
    var i: Int32 = 0
    while (i < 32):
        if (i >= MAX_ITERATIONS):
        var t: Float32 = (float(i) / 32.0)
        var pos: SIMD[DType.float32, 2] = (uv + (((direction * t) * len) * 0.1))
        var noise: Float32 = fbm(SIMD[DType.float32, 3]((pos * 10.0), (globals.scene.time * 0.05)), 4, 2.0, 0.5)
        var weight: Float32 = (1.0 - t)
        weight = (weight * weight)
        var noiseColor: SIMD[DType.float32, 3] = SIMD[DType.float32, 3]((0.5 + (0.5 * sin((((noise * 5.0) + globals.scene.time) + 0.0)))), (0.5 + (0.5 * sin((((noise * 5.0) + globals.scene.time) + 2.0)))), (0.5 + (0.5 * sin((((noise * 5.0) + globals.scene.time) + 4.0)))))
        color.rgb += (noiseColor * weight)
        totalWeight += weight
        direction = (mat2(cos((t * 3.0)), (-sin((t * 3.0))), sin((t * 3.0)), cos((t * 3.0))) * direction)
        (++i)
    color.rgb /= totalWeight
    color.a = 1.0
    var vignette: Float32 = (1.0 - smoothstep(0.5, 1.0, (magnitude((uv - 0.5)) * 1.5)))
    color.rgb *= vignette
    imageStore(outputImage, texCoord, color)

