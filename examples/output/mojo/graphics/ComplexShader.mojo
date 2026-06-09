# Generated Mojo Shader Code
from math import *
from gpu import *

# CrossGL resource placeholders
@value
struct Image2DFloat4:
    var width: Int32
    var height: Int32
    var depth_or_layers: Int32
    var levels: Int32
    var samples: Int32
    fn __init__(inout self, width: Int32 = 0, height: Int32 = 0, depth_or_layers: Int32 = 0, levels: Int32 = 1, samples: Int32 = 1):
        self.width = width
        self.height = height
        self.depth_or_layers = depth_or_layers
        self.levels = levels
        self.samples = samples

@value
struct Texture2D:
    var width: Int32
    var height: Int32
    var depth_or_layers: Int32
    var levels: Int32
    var samples: Int32
    fn __init__(inout self, width: Int32 = 0, height: Int32 = 0, depth_or_layers: Int32 = 0, levels: Int32 = 1, samples: Int32 = 1):
        self.width = width
        self.height = height
        self.depth_or_layers = depth_or_layers
        self.levels = levels
        self.samples = samples

fn sample(tex: Texture2D, coord: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

fn image_store(image: Image2DFloat4, coord: SIMD[DType.int32, 2], value: SIMD[DType.float32, 4]):
    pass


# CrossGL math helpers
fn clamp(value: Float32, min_value: Float32, max_value: Float32) -> Float32:
    return min(max(value, min_value), max_value)

fn clamp(value: SIMD[DType.float32, 2], min_value: Float32, max_value: Float32) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](clamp(value[0], min_value, max_value), clamp(value[1], min_value, max_value))

fn clamp(value: SIMD[DType.float32, 2], min_value: SIMD[DType.float32, 2], max_value: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](clamp(value[0], min_value[0], max_value[0]), clamp(value[1], min_value[1], max_value[1]))

fn clamp(value: SIMD[DType.float32, 4], min_value: Float32, max_value: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](clamp(value[0], min_value, max_value), clamp(value[1], min_value, max_value), clamp(value[2], min_value, max_value), clamp(value[3], min_value, max_value))

fn clamp(value: SIMD[DType.float32, 4], min_value: SIMD[DType.float32, 4], max_value: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](clamp(value[0], min_value[0], max_value[0]), clamp(value[1], min_value[1], max_value[1]), clamp(value[2], min_value[2], max_value[2]), clamp(value[3], min_value[3], max_value[3]))

fn dot_product(a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]) -> Float32:
    return a[0] * b[0] + a[1] * b[1]

fn dot_product(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> Float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

fn lerp(a: Float32, b: Float32, t: Float32) -> Float32:
    return a + (b - a) * t

fn lerp(a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2], t: Float32) -> SIMD[DType.float32, 2]:
    return a + (b - a) * t

fn lerp(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4], t: Float32) -> SIMD[DType.float32, 4]:
    return a + (b - a) * t

fn magnitude(v: SIMD[DType.float32, 2]) -> Float32:
    return sqrt(dot_product(v, v))

fn magnitude(v: SIMD[DType.float32, 4]) -> Float32:
    return sqrt(dot_product(v, v))

fn max(a: Float32, b: Float32) -> Float32:
    return a if a >= b else b

fn max(a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](a[0] if a[0] >= b[0] else b[0], a[1] if a[1] >= b[1] else b[1])

fn max(a: SIMD[DType.float32, 2], b: Float32) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](a[0] if a[0] >= b else b, a[1] if a[1] >= b else b)

fn max(a: Float32, b: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](a if a >= b[0] else b[0], a if a >= b[1] else b[1])

fn max(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a[0] if a[0] >= b[0] else b[0], a[1] if a[1] >= b[1] else b[1], a[2] if a[2] >= b[2] else b[2], a[3] if a[3] >= b[3] else b[3])

fn max(a: SIMD[DType.float32, 4], b: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a[0] if a[0] >= b else b, a[1] if a[1] >= b else b, a[2] if a[2] >= b else b, a[3] if a[3] >= b else b)

fn max(a: Float32, b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a if a >= b[0] else b[0], a if a >= b[1] else b[1], a if a >= b[2] else b[2], a if a >= b[3] else b[3])

fn min(a: Float32, b: Float32) -> Float32:
    return a if a <= b else b

fn min(a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](a[0] if a[0] <= b[0] else b[0], a[1] if a[1] <= b[1] else b[1])

fn min(a: SIMD[DType.float32, 2], b: Float32) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](a[0] if a[0] <= b else b, a[1] if a[1] <= b else b)

fn min(a: Float32, b: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](a if a <= b[0] else b[0], a if a <= b[1] else b[1])

fn min(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a[0] if a[0] <= b[0] else b[0], a[1] if a[1] <= b[1] else b[1], a[2] if a[2] <= b[2] else b[2], a[3] if a[3] <= b[3] else b[3])

fn min(a: SIMD[DType.float32, 4], b: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a[0] if a[0] <= b else b, a[1] if a[1] <= b else b, a[2] if a[2] <= b else b, a[3] if a[3] <= b else b)

fn min(a: Float32, b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a if a <= b[0] else b[0], a if a <= b[1] else b[1], a if a <= b[2] else b[2], a if a <= b[3] else b[3])

fn normalize(v: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    var len = magnitude(v)
    if len == 0.0:
        return v
    return v / len

fn normalize(v: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    var len = magnitude(v)
    if len == 0.0:
        return v
    return v / len

fn power(a: Float32, b: Float32) -> Float32:
    return pow(a, b)

fn power(a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](pow(a[0], b[0]), pow(a[1], b[1]))

fn power(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](pow(a[0], b[0]), pow(a[1], b[1]), pow(a[2], b[2]), pow(a[3], b[3]))

fn smoothstep(edge0: Float32, edge1: Float32, x: Float32) -> Float32:
    var t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

fn smoothstep(edge0: Float32, edge1: Float32, x: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](smoothstep(edge0, edge1, x[0]), smoothstep(edge0, edge1, x[1]))

fn smoothstep(edge0: SIMD[DType.float32, 2], edge1: SIMD[DType.float32, 2], x: Float32) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](smoothstep(edge0[0], edge1[0], x), smoothstep(edge0[1], edge1[1], x))

fn smoothstep(edge0: SIMD[DType.float32, 2], edge1: SIMD[DType.float32, 2], x: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](smoothstep(edge0[0], edge1[0], x[0]), smoothstep(edge0[1], edge1[1], x[1]))

fn smoothstep(edge0: Float32, edge1: Float32, x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](smoothstep(edge0, edge1, x[0]), smoothstep(edge0, edge1, x[1]), smoothstep(edge0, edge1, x[2]), smoothstep(edge0, edge1, x[3]))

fn smoothstep(edge0: SIMD[DType.float32, 4], edge1: SIMD[DType.float32, 4], x: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](smoothstep(edge0[0], edge1[0], x), smoothstep(edge0[1], edge1[1], x), smoothstep(edge0[2], edge1[2], x), smoothstep(edge0[3], edge1[3], x))

fn smoothstep(edge0: SIMD[DType.float32, 4], edge1: SIMD[DType.float32, 4], x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](smoothstep(edge0[0], edge1[0], x[0]), smoothstep(edge0[1], edge1[1], x[1]), smoothstep(edge0[2], edge1[2], x[2]), smoothstep(edge0[3], edge1[3], x[3]))

fn _crossgl_fract_f32(x: Float32) -> Float32:
    return x - floor(x)

fn _crossgl_fract_f32_3_4(v: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](v[0] - floor(v[0]), v[1] - floor(v[1]), v[2] - floor(v[2]), 0.0)


# CrossGL matrix types
@value
struct CrossGLMatrixF32C2R2:
    var c0: SIMD[DType.float32, 2]
    var c1: SIMD[DType.float32, 2]

    fn __getitem__(self, index: Int) -> SIMD[DType.float32, 2]:
        if index == 0:
            return self.c0
        return self.c1

    fn __setitem__(inout self, index: Int, value: SIMD[DType.float32, 2]):
        if index == 0:
            self.c0 = value
            return
        self.c1 = value

    fn __getitem__(self, index: Int32) -> SIMD[DType.float32, 2]:
        if index == 0:
            return self.c0
        return self.c1

    fn __setitem__(inout self, index: Int32, value: SIMD[DType.float32, 2]):
        if index == 0:
            self.c0 = value
            return
        self.c1 = value

    fn __getitem__(self, index: UInt32) -> SIMD[DType.float32, 2]:
        if index == 0:
            return self.c0
        return self.c1

    fn __setitem__(inout self, index: UInt32, value: SIMD[DType.float32, 2]):
        if index == 0:
            self.c0 = value
            return
        self.c1 = value

    fn __mul__(self, value: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
        return self.c0 * value[0] + self.c1 * value[1]

    fn __mul__(self, other: CrossGLMatrixF32C2R2) -> CrossGLMatrixF32C2R2:
        return CrossGLMatrixF32C2R2(self * other.c0, self * other.c1)

fn transpose(m: CrossGLMatrixF32C2R2) -> CrossGLMatrixF32C2R2:
    return CrossGLMatrixF32C2R2(SIMD[DType.float32, 2](m.c0[0], m.c1[0]), SIMD[DType.float32, 2](m.c0[1], m.c1[1]))

fn inverse(m: CrossGLMatrixF32C2R2) -> CrossGLMatrixF32C2R2:
    return m


@value
struct CrossGLMatrixF32C3R3:
    var c0: SIMD[DType.float32, 4]
    var c1: SIMD[DType.float32, 4]
    var c2: SIMD[DType.float32, 4]

    fn __getitem__(self, index: Int) -> SIMD[DType.float32, 4]:
        if index == 0:
            return self.c0
        if index == 1:
            return self.c1
        return self.c2

    fn __setitem__(inout self, index: Int, value: SIMD[DType.float32, 4]):
        if index == 0:
            self.c0 = value
            return
        if index == 1:
            self.c1 = value
            return
        self.c2 = value

    fn __getitem__(self, index: Int32) -> SIMD[DType.float32, 4]:
        if index == 0:
            return self.c0
        if index == 1:
            return self.c1
        return self.c2

    fn __setitem__(inout self, index: Int32, value: SIMD[DType.float32, 4]):
        if index == 0:
            self.c0 = value
            return
        if index == 1:
            self.c1 = value
            return
        self.c2 = value

    fn __getitem__(self, index: UInt32) -> SIMD[DType.float32, 4]:
        if index == 0:
            return self.c0
        if index == 1:
            return self.c1
        return self.c2

    fn __setitem__(inout self, index: UInt32, value: SIMD[DType.float32, 4]):
        if index == 0:
            self.c0 = value
            return
        if index == 1:
            self.c1 = value
            return
        self.c2 = value

    fn __mul__(self, value: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
        return self.c0 * value[0] + self.c1 * value[1] + self.c2 * value[2]

    fn __mul__(self, other: CrossGLMatrixF32C3R3) -> CrossGLMatrixF32C3R3:
        return CrossGLMatrixF32C3R3(self * other.c0, self * other.c1, self * other.c2)

fn transpose(m: CrossGLMatrixF32C3R3) -> CrossGLMatrixF32C3R3:
    return CrossGLMatrixF32C3R3(SIMD[DType.float32, 4](m.c0[0], m.c1[0], m.c2[0], 0.0), SIMD[DType.float32, 4](m.c0[1], m.c1[1], m.c2[1], 0.0), SIMD[DType.float32, 4](m.c0[2], m.c1[2], m.c2[2], 0.0))

fn inverse(m: CrossGLMatrixF32C3R3) -> CrossGLMatrixF32C3R3:
    return m


@value
struct CrossGLMatrixF32C4R4:
    var c0: SIMD[DType.float32, 4]
    var c1: SIMD[DType.float32, 4]
    var c2: SIMD[DType.float32, 4]
    var c3: SIMD[DType.float32, 4]

    fn __getitem__(self, index: Int) -> SIMD[DType.float32, 4]:
        if index == 0:
            return self.c0
        if index == 1:
            return self.c1
        if index == 2:
            return self.c2
        return self.c3

    fn __setitem__(inout self, index: Int, value: SIMD[DType.float32, 4]):
        if index == 0:
            self.c0 = value
            return
        if index == 1:
            self.c1 = value
            return
        if index == 2:
            self.c2 = value
            return
        self.c3 = value

    fn __getitem__(self, index: Int32) -> SIMD[DType.float32, 4]:
        if index == 0:
            return self.c0
        if index == 1:
            return self.c1
        if index == 2:
            return self.c2
        return self.c3

    fn __setitem__(inout self, index: Int32, value: SIMD[DType.float32, 4]):
        if index == 0:
            self.c0 = value
            return
        if index == 1:
            self.c1 = value
            return
        if index == 2:
            self.c2 = value
            return
        self.c3 = value

    fn __getitem__(self, index: UInt32) -> SIMD[DType.float32, 4]:
        if index == 0:
            return self.c0
        if index == 1:
            return self.c1
        if index == 2:
            return self.c2
        return self.c3

    fn __setitem__(inout self, index: UInt32, value: SIMD[DType.float32, 4]):
        if index == 0:
            self.c0 = value
            return
        if index == 1:
            self.c1 = value
            return
        if index == 2:
            self.c2 = value
            return
        self.c3 = value

    fn __mul__(self, value: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
        return self.c0 * value[0] + self.c1 * value[1] + self.c2 * value[2] + self.c3 * value[3]

    fn __mul__(self, other: CrossGLMatrixF32C4R4) -> CrossGLMatrixF32C4R4:
        return CrossGLMatrixF32C4R4(self * other.c0, self * other.c1, self * other.c2, self * other.c3)

fn transpose(m: CrossGLMatrixF32C4R4) -> CrossGLMatrixF32C4R4:
    return CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](m.c0[0], m.c1[0], m.c2[0], m.c3[0]), SIMD[DType.float32, 4](m.c0[1], m.c1[1], m.c2[1], m.c3[1]), SIMD[DType.float32, 4](m.c0[2], m.c1[2], m.c2[2], m.c3[2]), SIMD[DType.float32, 4](m.c0[3], m.c1[3], m.c2[3], m.c3[3]))

fn inverse(m: CrossGLMatrixF32C4R4) -> CrossGLMatrixF32C4R4:
    return m



# CrossGL vector helpers
fn _crossgl_vec3_mul_f32_sv(s: Float32, v: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](s * v[0], s * v[1], s * v[2], 0.0)

fn _crossgl_vec3_mul_f32_vs(v: SIMD[DType.float32, 4], s: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](v[0] * s, v[1] * s, v[2] * s, 0.0)

fn _crossgl_vec3_mul_f32_vv(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a[0] * b[0], a[1] * b[1], a[2] * b[2], 0.0)

fn _crossgl_vec3_add_f32_vs(v: SIMD[DType.float32, 4], s: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](v[0] + s, v[1] + s, v[2] + s, 0.0)

fn _crossgl_vec3_add_f32_vv(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a[0] + b[0], a[1] + b[1], a[2] + b[2], 0.0)

fn _crossgl_vec3_sub_f32_sv(s: Float32, v: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](s - v[0], s - v[1], s - v[2], 0.0)

fn _crossgl_vec3_sub_f32_vs(v: SIMD[DType.float32, 4], s: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](v[0] - s, v[1] - s, v[2] - s, 0.0)

fn _crossgl_vec3_sub_f32_vv(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a[0] - b[0], a[1] - b[1], a[2] - b[2], 0.0)

fn _crossgl_vec3_div_f32_vs(v: SIMD[DType.float32, 4], s: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](v[0] / s, v[1] / s, v[2] / s, 0.0)

fn _crossgl_vec3_div_f32_vv(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a[0] / b[0], a[1] / b[1], a[2] / b[2], 0.0)

fn _crossgl_vec3_splat_f32(s: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](s, s, s, 0.0)

fn _crossgl_construct_f32_4_vf322_01_s(v0: SIMD[DType.float32, 2], s1: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](v0[0], v0[1], s1, 0.0)

fn _crossgl_construct_f32_4_vf324_012_s(v0: SIMD[DType.float32, 4], s1: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](v0[0], v0[1], v0[2], s1)


@value
struct Material:
    var albedo: SIMD[DType.float32, 4]
    var roughness: Float32
    var metallic: Float32
    var emissive: SIMD[DType.float32, 4]
    var opacity: Float32
    var hasNormalMap: Bool
    var albedoMap: Texture2D
    var normalMap: Texture2D
    var metallicRoughnessMap: Texture2D

@value
struct Light:
    var position: SIMD[DType.float32, 4]
    var color: SIMD[DType.float32, 4]
    var intensity: Float32
    var radius: Float32
    var castShadows: Bool
    var viewProjection: CrossGLMatrixF32C4R4

@value
struct Scene:
    var materials: InlineArray[Material, 4]
    var lights: InlineArray[Light, 8]
    var ambientLight: SIMD[DType.float32, 4]
    var time: Float32
    var elapsedTime: Float32
    var activeLightCount: Int32
    var viewMatrix: CrossGLMatrixF32C4R4
    var projectionMatrix: CrossGLMatrixF32C4R4

@value
struct VertexInput:
    var position: SIMD[DType.float32, 4]
    var normal: SIMD[DType.float32, 4]
    var tangent: SIMD[DType.float32, 4]
    var bitangent: SIMD[DType.float32, 4]
    var texCoord0: SIMD[DType.float32, 2]
    var texCoord1: SIMD[DType.float32, 2]
    var color: SIMD[DType.float32, 4]
    var materialIndex: Int32

@value
struct VertexOutput:
    var worldPosition: SIMD[DType.float32, 4]
    var worldNormal: SIMD[DType.float32, 4]
    var worldTangent: SIMD[DType.float32, 4]
    var worldBitangent: SIMD[DType.float32, 4]
    var texCoord0: SIMD[DType.float32, 2]
    var texCoord1: SIMD[DType.float32, 2]
    var color: SIMD[DType.float32, 4]
    var TBN: CrossGLMatrixF32C3R3
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
    var cameraPosition: SIMD[DType.float32, 4]
    var globalRoughness: Float32
    var screenSize: SIMD[DType.float32, 2]
    var nearPlane: Float32
    var farPlane: Float32
    var frameCount: Int32
    var noiseValues: List[Float32]

alias PI = 3.14159265359
alias EPSILON = 0.0001
alias MAX_ITERATIONS = 64
alias UP_VECTOR = SIMD[DType.float32, 4](0.0, 1.0, 0.0, 0.0)

fn distributionGGX(N: SIMD[DType.float32, 4], H: SIMD[DType.float32, 4], roughness: Float32) -> Float32:
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

fn geometrySmith(N: SIMD[DType.float32, 4], V: SIMD[DType.float32, 4], L: SIMD[DType.float32, 4], roughness: Float32) -> Float32:
    var NdotV: Float32 = max(dot_product(N, V), 0.0)
    var NdotL: Float32 = max(dot_product(N, L), 0.0)
    var ggx2: Float32 = geometrySchlickGGX(NdotV, roughness)
    var ggx1: Float32 = geometrySchlickGGX(NdotL, roughness)
    return (ggx1 * ggx2)

fn fresnelSchlick(cosTheta: Float32, F0: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return _crossgl_vec3_add_f32_vv(F0, _crossgl_vec3_mul_f32_vs(_crossgl_vec3_sub_f32_sv(1.0, F0), power(max((1.0 - cosTheta), 0.0), 5.0)))

fn noise3D(p: SIMD[DType.float32, 4]) -> Float32:
    var i: SIMD[DType.float32, 4] = floor(p)
    var f: SIMD[DType.float32, 4] = _crossgl_fract_f32_3_4(p)
    var u: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_vv(_crossgl_vec3_mul_f32_vv(_crossgl_vec3_mul_f32_vv(f, f), f), _crossgl_vec3_add_f32_vs(_crossgl_vec3_mul_f32_vv(f, _crossgl_vec3_sub_f32_vs(_crossgl_vec3_mul_f32_vs(f, 6.0), 15.0)), 10.0))
    var n000: Float32 = _crossgl_fract_f32((sin(dot_product(i, SIMD[DType.float32, 4](13.534, 43.5234, 243.32, 0.0))) * 4453.0))
    var n001: Float32 = _crossgl_fract_f32((sin(dot_product(_crossgl_vec3_add_f32_vv(i, SIMD[DType.float32, 4](0.0, 0.0, 1.0, 0.0)), SIMD[DType.float32, 4](13.534, 43.5234, 243.32, 0.0))) * 4453.0))
    var n010: Float32 = _crossgl_fract_f32((sin(dot_product(_crossgl_vec3_add_f32_vv(i, SIMD[DType.float32, 4](0.0, 1.0, 0.0, 0.0)), SIMD[DType.float32, 4](13.534, 43.5234, 243.32, 0.0))) * 4453.0))
    var n011: Float32 = _crossgl_fract_f32((sin(dot_product(_crossgl_vec3_add_f32_vv(i, SIMD[DType.float32, 4](0.0, 1.0, 1.0, 0.0)), SIMD[DType.float32, 4](13.534, 43.5234, 243.32, 0.0))) * 4453.0))
    var n100: Float32 = _crossgl_fract_f32((sin(dot_product(_crossgl_vec3_add_f32_vv(i, SIMD[DType.float32, 4](1.0, 0.0, 0.0, 0.0)), SIMD[DType.float32, 4](13.534, 43.5234, 243.32, 0.0))) * 4453.0))
    var n101: Float32 = _crossgl_fract_f32((sin(dot_product(_crossgl_vec3_add_f32_vv(i, SIMD[DType.float32, 4](1.0, 0.0, 1.0, 0.0)), SIMD[DType.float32, 4](13.534, 43.5234, 243.32, 0.0))) * 4453.0))
    var n110: Float32 = _crossgl_fract_f32((sin(dot_product(_crossgl_vec3_add_f32_vv(i, SIMD[DType.float32, 4](1.0, 1.0, 0.0, 0.0)), SIMD[DType.float32, 4](13.534, 43.5234, 243.32, 0.0))) * 4453.0))
    var n111: Float32 = _crossgl_fract_f32((sin(dot_product(_crossgl_vec3_add_f32_vv(i, SIMD[DType.float32, 4](1.0, 1.0, 1.0, 0.0)), SIMD[DType.float32, 4](13.534, 43.5234, 243.32, 0.0))) * 4453.0))
    var n00: Float32 = lerp(n000, n001, u[2])
    var n01: Float32 = lerp(n010, n011, u[2])
    var n10: Float32 = lerp(n100, n101, u[2])
    var n11: Float32 = lerp(n110, n111, u[2])
    var n0: Float32 = lerp(n00, n01, u[1])
    var n1: Float32 = lerp(n10, n11, u[1])
    return lerp(n0, n1, u[0])

fn fbm(p: SIMD[DType.float32, 4], octaves: Int32, lacunarity: Float32, gain: Float32) -> Float32:
    var sum: Float32 = 0.0
    var amplitude: Float32 = 1.0
    var frequency: Float32 = 1.0
    var i: Int32 = 0
    while (i < octaves):
        if (i >= MAX_ITERATIONS):
            break
        sum += (amplitude * noise3D(_crossgl_vec3_mul_f32_vs(p, frequency)))
        amplitude *= gain
        frequency *= lacunarity
        i += 1
    return sum

fn samplePlanarProjection(tex: Texture2D, worldPos: SIMD[DType.float32, 4], normal: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    var absNormal: SIMD[DType.float32, 4] = abs(normal)
    var useX: Bool = ((absNormal[0] >= absNormal[1]) and (absNormal[0] >= absNormal[2]))
    var useY: Bool = ((not useX) and (absNormal[1] >= absNormal[2]))
    var uv: SIMD[DType.float32, 2]
    if useX:
        uv = ((SIMD[DType.float32, 2](worldPos[2], worldPos[1]) * 0.5) + 0.5)
        if (normal[0] < 0.0):
            uv[0] = (1.0 - uv[0])
    elif useY:
        uv = ((SIMD[DType.float32, 2](worldPos[0], worldPos[2]) * 0.5) + 0.5)
        if (normal[1] < 0.0):
            uv[1] = (1.0 - uv[1])
    else:
        uv = ((SIMD[DType.float32, 2](worldPos[0], worldPos[1]) * 0.5) + 0.5)
        if (normal[2] < 0.0):
            uv[0] = (1.0 - uv[0])
    return sample(tex, uv)

# Vertex Shader
var globals = GlobalUniforms(Scene(InlineArray[Material, 4](Material(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, False, Texture2D(), Texture2D(), Texture2D()), Material(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, False, Texture2D(), Texture2D(), Texture2D()), Material(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, False, Texture2D(), Texture2D(), Texture2D()), Material(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, False, Texture2D(), Texture2D(), Texture2D())), InlineArray[Light, 8](Light(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, False, CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))), Light(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, False, CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))), Light(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, False, CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))), Light(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, False, CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))), Light(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, False, CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))), Light(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, False, CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))), Light(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, False, CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))), Light(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, False, CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)))), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0, CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)), CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, SIMD[DType.float32, 2](0.0, 0.0), 0.0, 0.0, 0, List[Float32]())
# CrossGL shader stage: vertex
fn vertex_main(input: VertexInput) -> VertexOutput:
    var output = VertexOutput(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 2](0.0, 0.0), SIMD[DType.float32, 2](0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), CrossGLMatrixF32C3R3(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)), 0, SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
    var modelMatrix: CrossGLMatrixF32C4R4 = CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](1.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 1.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 1.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 1.0))
    var viewMatrix: CrossGLMatrixF32C4R4 = globals.scene.viewMatrix
    var projectionMatrix: CrossGLMatrixF32C4R4 = globals.scene.projectionMatrix
    var modelViewMatrix: CrossGLMatrixF32C4R4 = (viewMatrix * modelMatrix)
    var modelViewProjectionMatrix: CrossGLMatrixF32C4R4 = (projectionMatrix * modelViewMatrix)
    var normalMatrix: CrossGLMatrixF32C3R3 = CrossGLMatrixF32C3R3(SIMD[DType.float32, 4](transpose(inverse(modelMatrix)).c0[0], transpose(inverse(modelMatrix)).c0[1], transpose(inverse(modelMatrix)).c0[2], 0.0), SIMD[DType.float32, 4](transpose(inverse(modelMatrix)).c1[0], transpose(inverse(modelMatrix)).c1[1], transpose(inverse(modelMatrix)).c1[2], 0.0), SIMD[DType.float32, 4](transpose(inverse(modelMatrix)).c2[0], transpose(inverse(modelMatrix)).c2[1], transpose(inverse(modelMatrix)).c2[2], 0.0))
    var worldPosition: SIMD[DType.float32, 4] = (modelMatrix * SIMD[DType.float32, 4](input.position[0], input.position[1], input.position[2], 1.0))
    var worldNormal: SIMD[DType.float32, 4] = normalize((normalMatrix * input.normal))
    var worldTangent: SIMD[DType.float32, 4] = normalize((normalMatrix * input.tangent))
    var worldBitangent: SIMD[DType.float32, 4] = normalize((normalMatrix * input.bitangent))
    var TBN: CrossGLMatrixF32C3R3 = CrossGLMatrixF32C3R3(SIMD[DType.float32, 4](worldTangent[0], worldTangent[1], worldTangent[2], 0.0), SIMD[DType.float32, 4](worldBitangent[0], worldBitangent[1], worldBitangent[2], 0.0), SIMD[DType.float32, 4](worldNormal[0], worldNormal[1], worldNormal[2], 0.0))
    var displacement: Float32 = (fbm(_crossgl_vec3_add_f32_vs(SIMD[DType.float32, 4](worldPosition[0], worldPosition[1], worldPosition[2], 0.0), (globals.scene.time * 0.1)), 4, 2.0, 0.5) * 0.1)
    if (input.materialIndex > 0):
        var __cgl_swizzle_0: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_vs(worldNormal, displacement)
        worldPosition[0] += __cgl_swizzle_0[0]
        worldPosition[1] += __cgl_swizzle_0[1]
        worldPosition[2] += __cgl_swizzle_0[2]
    var viewDir: SIMD[DType.float32, 4] = normalize(_crossgl_vec3_sub_f32_vv(globals.cameraPosition, SIMD[DType.float32, 4](worldPosition[0], worldPosition[1], worldPosition[2], 0.0)))
    var fresnel: Float32 = power((1.0 - max(0.0, dot_product(worldNormal, viewDir))), 5.0)
    if (input.materialIndex < globals.scene.activeLightCount):
        output.color = (input.color * SIMD[DType.float32, 4](1.0, 1.0, 1.0, 1.0))
        var i: Int32 = 0
        while (i < 4):
            if (i >= (globals.frameCount % 5)):
                break
            var light: Light = globals.scene.lights[int(i)]
            var lightDir: SIMD[DType.float32, 4] = normalize(_crossgl_vec3_sub_f32_vv(light.position, SIMD[DType.float32, 4](worldPosition[0], worldPosition[1], worldPosition[2], 0.0)))
            var lightDistance: Float32 = magnitude(_crossgl_vec3_sub_f32_vv(light.position, SIMD[DType.float32, 4](worldPosition[0], worldPosition[1], worldPosition[2], 0.0)))
            var attenuation: Float32 = (1.0 / (1.0 + (lightDistance * lightDistance)))
            var lightIntensity: Float32 = (light.intensity * attenuation)
            var __cgl_swizzle_1: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_vs(_crossgl_vec3_mul_f32_vs(_crossgl_vec3_mul_f32_vs(light.color, lightIntensity), max(0.0, dot_product(worldNormal, lightDir))), 0.025)
            output.color[0] += __cgl_swizzle_1[0]
            output.color[1] += __cgl_swizzle_1[1]
            output.color[2] += __cgl_swizzle_1[2]
            i += 1
    else:
        output.color = input.color
        if (globals.globalRoughness > 0.5):
            if (fresnel > 0.7):
                output.color[3] *= 0.8
            else:
                output.color[3] *= 0.9
    output.worldPosition = SIMD[DType.float32, 4](worldPosition[0], worldPosition[1], worldPosition[2], 0.0)
    output.worldNormal = worldNormal
    output.worldTangent = worldTangent
    output.worldBitangent = worldBitangent
    output.texCoord0 = input.texCoord0
    output.texCoord1 = input.texCoord1
    output.TBN = TBN
    output.materialIndex = input.materialIndex
    output.clipPosition = (modelViewProjectionMatrix * SIMD[DType.float32, 4](input.position[0], input.position[1], input.position[2], 1.0))
    return output

# Fragment Shader
# CrossGL resource metadata: name=shadowMap kind=texture set=0 binding=0 binding_source=automatic
var shadowMap: Texture2D = Texture2D()
fn shadowCalculation(fragPosLightSpace: SIMD[DType.float32, 4], iteration: Int32, input: VertexOutput) -> Float32:
    if (iteration > 3):
        return 0.0
    var projCoords: SIMD[DType.float32, 4] = _crossgl_vec3_div_f32_vs(SIMD[DType.float32, 4](fragPosLightSpace[0], fragPosLightSpace[1], fragPosLightSpace[2], 0.0), fragPosLightSpace[3])
    projCoords = _crossgl_vec3_add_f32_vs(_crossgl_vec3_mul_f32_vs(projCoords, 0.5), 0.5)
    var closestDepth: Float32 = sample(shadowMap, SIMD[DType.float32, 2](projCoords[0], projCoords[1]))[0]
    var currentDepth: Float32 = projCoords[2]
    var bias: Float32 = max((0.05 * (1.0 - dot_product(input.worldNormal, normalize(_crossgl_vec3_sub_f32_vv(globals.cameraPosition, input.worldPosition))))), 0.005)
    var shadow: Float32 = (1.0 if ((currentDepth - bias) > closestDepth) else 0.0)
    var pcfDepth: Float32 = 0.0
    var texelSize: SIMD[DType.float32, 2] = (1.0 / SIMD[DType.float32, 2](globals.screenSize[0], globals.screenSize[1]))
    var offset: Float32 = (globals.noiseValues[int(((iteration * 4) % 16))] * 0.001)
    var x: Int32 = (-1)
    while (x <= 1):
        var y: Int32 = (-1)
        while (y <= 1):
            var pcfDepth: Float32 = sample(shadowMap, ((SIMD[DType.float32, 2](projCoords[0], projCoords[1]) + (SIMD[DType.float32, 2]((x).cast[DType.float32](), (y).cast[DType.float32]()) * texelSize)) + SIMD[DType.float32, 2](offset)))[0]
            shadow += (1.0 if ((currentDepth - bias) > pcfDepth) else 0.0)
            y += 1
        x += 1
    shadow /= 9.0
    if (projCoords[2] > 1.0):
        shadow = 0.0
    return shadow

# CrossGL shader stage: fragment
fn fragment_main(input: VertexOutput) -> FragmentOutput:
    var output = FragmentOutput(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0)
    var material: Material = globals.scene.materials[int(input.materialIndex)]
    var albedoValue: SIMD[DType.float32, 4] = sample(material.albedoMap, input.texCoord0)
    var normalValue: SIMD[DType.float32, 4] = sample(material.normalMap, input.texCoord0)
    var metallicRoughnessValue: SIMD[DType.float32, 4] = sample(material.metallicRoughnessMap, input.texCoord0)
    var normal: SIMD[DType.float32, 4] = _crossgl_vec3_sub_f32_vs(_crossgl_vec3_mul_f32_vs(SIMD[DType.float32, 4](normalValue[0], normalValue[1], normalValue[2], 0.0), 2.0), 1.0)
    var worldNormal: SIMD[DType.float32, 4] = normalize((input.TBN * normal))
    var albedo: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_vv(SIMD[DType.float32, 4](albedoValue[0], albedoValue[1], albedoValue[2], 0.0), material.albedo)
    var metallic: Float32 = (metallicRoughnessValue[2] * material.metallic)
    var roughness: Float32 = (metallicRoughnessValue[1] * material.roughness)
    var ao: Float32 = metallicRoughnessValue[0]
    var viewDir: SIMD[DType.float32, 4] = normalize(_crossgl_vec3_sub_f32_vv(globals.cameraPosition, input.worldPosition))
    var F0: SIMD[DType.float32, 4] = lerp(SIMD[DType.float32, 4](0.04, 0.04, 0.04, 0.0), albedo, metallic)
    var Lo: SIMD[DType.float32, 4] = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
    var i: Int32 = 0
    while (i < globals.scene.activeLightCount):
        if (i >= 8):
            break
        var light: Light = globals.scene.lights[int(i)]
        var lightDir: SIMD[DType.float32, 4] = normalize(_crossgl_vec3_sub_f32_vv(light.position, input.worldPosition))
        var halfway: SIMD[DType.float32, 4] = normalize(_crossgl_vec3_add_f32_vv(viewDir, lightDir))
        var distance: Float32 = magnitude(_crossgl_vec3_sub_f32_vv(light.position, input.worldPosition))
        var attenuation: Float32 = (1.0 / (distance * distance))
        var radiance: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_vs(_crossgl_vec3_mul_f32_vs(light.color, light.intensity), attenuation)
        var NDF: Float32 = distributionGGX(worldNormal, halfway, roughness)
        var G: Float32 = geometrySmith(worldNormal, viewDir, lightDir, roughness)
        var F: SIMD[DType.float32, 4] = fresnelSchlick(max(dot_product(halfway, viewDir), 0.0), F0)
        var kS: SIMD[DType.float32, 4] = F
        var kD: SIMD[DType.float32, 4] = _crossgl_vec3_sub_f32_vv(SIMD[DType.float32, 4](1.0, 1.0, 1.0, 0.0), kS)
        kD *= (1.0 - metallic)
        var numerator: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_sv((NDF * G), F)
        var denominator: Float32 = (((4.0 * max(dot_product(worldNormal, viewDir), 0.0)) * max(dot_product(worldNormal, lightDir), 0.0)) + EPSILON)
        var specular: SIMD[DType.float32, 4] = _crossgl_vec3_div_f32_vs(numerator, denominator)
        var NdotL: Float32 = max(dot_product(worldNormal, lightDir), 0.0)
        var shadow: Float32 = 0.0
        if light.castShadows:
            var fragPosLightSpace: SIMD[DType.float32, 4] = (light.viewProjection * SIMD[DType.float32, 4](input.worldPosition[0], input.worldPosition[1], input.worldPosition[2], 1.0))
            shadow = shadowCalculation(fragPosLightSpace, 0, input)
            var s: Int32 = 0
            while (s < 4):
                if (s >= (globals.frameCount % 3)):
                    s += 1
                    continue
                shadow += shadowCalculation((fragPosLightSpace + SIMD[DType.float32, 4]((globals.noiseValues[int((s % 16))] * 0.001), 0.0, 0.0, 0.0)), (s + 1), input)
                s += 1
            shadow /= 5.0
        Lo += _crossgl_vec3_mul_f32_vs(_crossgl_vec3_mul_f32_vv(_crossgl_vec3_mul_f32_sv((1.0 - shadow), _crossgl_vec3_add_f32_vv(_crossgl_vec3_div_f32_vs(_crossgl_vec3_mul_f32_vv(kD, albedo), PI), specular)), radiance), NdotL)
        i += 1
    var ambient: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_vs(_crossgl_vec3_mul_f32_vv(globals.scene.ambientLight, albedo), ao)
    var color: SIMD[DType.float32, 4] = _crossgl_vec3_add_f32_vv(ambient, Lo)
    color = _crossgl_vec3_div_f32_vv(color, _crossgl_vec3_add_f32_vv(color, SIMD[DType.float32, 4](1.0, 1.0, 1.0, 0.0)))
    color = power(color, _crossgl_vec3_splat_f32((1.0 / 2.2)))
    output.color = SIMD[DType.float32, 4](color[0], color[1], color[2], (material.opacity * albedoValue[3]))
    output.normalBuffer = _crossgl_construct_f32_4_vf324_012_s(_crossgl_vec3_add_f32_vs(_crossgl_vec3_mul_f32_vs(worldNormal, 0.5), 0.5), 1.0)
    output.positionBuffer = SIMD[DType.float32, 4](input.worldPosition[0], input.worldPosition[1], input.worldPosition[2], 1.0)
    output.depth = (input.clipPosition[2] / input.clipPosition[3])
    return output

# Compute Shader
# CrossGL resource metadata: name=outputImage kind=image set=0 binding=0 binding_source=automatic
var outputImage: Image2DFloat4 = Image2DFloat4()
# CrossGL shader stage: compute
fn compute_main() -> None:
    var texCoord: SIMD[DType.int32, 2] = SIMD[DType.int32, 2](SIMD[DType.uint32, 4](global_idx_uint.x, global_idx_uint.y, global_idx_uint.z, 0)[0].cast[DType.int32](), SIMD[DType.uint32, 4](global_idx_uint.x, global_idx_uint.y, global_idx_uint.z, 0)[1].cast[DType.int32]())
    var screenSize: SIMD[DType.float32, 2] = globals.screenSize
    if ((texCoord[0] >= Int32(screenSize[0])) or (texCoord[1] >= Int32(screenSize[1]))):
        return None
    var uv: SIMD[DType.float32, 2] = (SIMD[DType.float32, 2](texCoord[0].cast[DType.float32](), texCoord[1].cast[DType.float32]()) / screenSize)
    var color: SIMD[DType.float32, 4] = SIMD[DType.float32, 4](0.0)
    var totalWeight: Float32 = 0.0
    var direction: SIMD[DType.float32, 2] = (SIMD[DType.float32, 2](0.5) - uv)
    var len: Float32 = magnitude(direction)
    direction = normalize(direction)
    var i: Int32 = 0
    while (i < 32):
        if (i >= MAX_ITERATIONS):
            break
        var t: Float32 = (Float32(i) / 32.0)
        var pos: SIMD[DType.float32, 2] = (uv + (((direction * t) * len) * 0.1))
        var noise: Float32 = fbm(_crossgl_construct_f32_4_vf322_01_s((pos * 10.0), (globals.scene.time * 0.05)), 4, 2.0, 0.5)
        var weight: Float32 = (1.0 - t)
        weight = (weight * weight)
        var noiseColor: SIMD[DType.float32, 4] = SIMD[DType.float32, 4]((0.5 + (0.5 * sin((((noise * 5.0) + globals.scene.time) + 0.0)))), (0.5 + (0.5 * sin((((noise * 5.0) + globals.scene.time) + 2.0)))), (0.5 + (0.5 * sin((((noise * 5.0) + globals.scene.time) + 4.0)))), 0.0)
        var __cgl_swizzle_2: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_vs(noiseColor, weight)
        color[0] += __cgl_swizzle_2[0]
        color[1] += __cgl_swizzle_2[1]
        color[2] += __cgl_swizzle_2[2]
        totalWeight += weight
        direction = (CrossGLMatrixF32C2R2(SIMD[DType.float32, 2](cos((t * 3.0)), (-sin((t * 3.0)))), SIMD[DType.float32, 2](sin((t * 3.0)), cos((t * 3.0)))) * direction)
        i += 1
    var __cgl_swizzle_3: SIMD[DType.float32, 4] = totalWeight
    color[0] /= __cgl_swizzle_3[0]
    color[1] /= __cgl_swizzle_3[1]
    color[2] /= __cgl_swizzle_3[2]
    color[3] = 1.0
    var vignette: Float32 = (1.0 - smoothstep(0.5, 1.0, (magnitude((uv - 0.5)) * 1.5)))
    var __cgl_swizzle_4: SIMD[DType.float32, 4] = vignette
    color[0] *= __cgl_swizzle_4[0]
    color[1] *= __cgl_swizzle_4[1]
    color[2] *= __cgl_swizzle_4[2]
    image_store(outputImage, texCoord, color)

