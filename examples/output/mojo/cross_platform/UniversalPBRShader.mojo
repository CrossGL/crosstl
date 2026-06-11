# Generated Mojo Shader Code
from math import *
from gpu import *

# CrossGL GPU builtin placeholders
@value
struct _CrossGLGpuBuiltinU32Vec3:
    var x: UInt32
    var y: UInt32
    var z: UInt32

var global_idx_uint = _CrossGLGpuBuiltinU32Vec3(0, 0, 0)

# CrossGL resource placeholders
@value
struct ImageCubeFloat4:
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

@value
struct TextureCube:
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

fn sample(tex: TextureCube, coord: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

fn sample_lod(tex: TextureCube, coord: SIMD[DType.float32, 4], lod: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](0.0, 0.0, lod, 1.0)

fn _crossgl_mip_dimension(value: Int32, lod: Int32) -> Int32:
    if value <= 0:
        return 0
    var scaled = value
    var level = lod
    while level > 0:
        if scaled > 1:
            scaled = scaled >> 1
        level -= 1
    return scaled

fn image_size(image: ImageCubeFloat4) -> SIMD[DType.int32, 2]:
    return SIMD[DType.int32, 2](image.width, image.height)

fn texture_size(tex: Texture2D) -> SIMD[DType.int32, 2]:
    return SIMD[DType.int32, 2](tex.width, tex.height)

fn texture_size(tex: Texture2D, lod: Int32) -> SIMD[DType.int32, 2]:
    return SIMD[DType.int32, 2](_crossgl_mip_dimension(tex.width, lod), _crossgl_mip_dimension(tex.height, lod))

fn image_store(image: ImageCubeFloat4, coord: SIMD[DType.int32, 4], value: SIMD[DType.float32, 4]):
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

fn cross_product(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0.0)

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

fn max(a: Int8, b: Int8) -> Int8:
    return a if a >= b else b

fn max(a: Int16, b: Int16) -> Int16:
    return a if a >= b else b

fn max(a: Int32, b: Int32) -> Int32:
    return a if a >= b else b

fn max(a: Int64, b: Int64) -> Int64:
    return a if a >= b else b

fn max(a: UInt8, b: UInt8) -> UInt8:
    return a if a >= b else b

fn max(a: UInt16, b: UInt16) -> UInt16:
    return a if a >= b else b

fn max(a: UInt32, b: UInt32) -> UInt32:
    return a if a >= b else b

fn max(a: UInt64, b: UInt64) -> UInt64:
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

fn min(a: Int8, b: Int8) -> Int8:
    return a if a <= b else b

fn min(a: Int16, b: Int16) -> Int16:
    return a if a <= b else b

fn min(a: Int32, b: Int32) -> Int32:
    return a if a <= b else b

fn min(a: Int64, b: Int64) -> Int64:
    return a if a <= b else b

fn min(a: UInt8, b: UInt8) -> UInt8:
    return a if a <= b else b

fn min(a: UInt16, b: UInt16) -> UInt16:
    return a if a <= b else b

fn min(a: UInt32, b: UInt32) -> UInt32:
    return a if a <= b else b

fn min(a: UInt64, b: UInt64) -> UInt64:
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

fn reflect(i: SIMD[DType.float32, 4], n: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return i - n * (2.0 * dot_product(n, i))

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


# CrossGL matrix types
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

fn _crossgl_swizzle_f32_4_rg(v: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](v[0], v[1])

fn _crossgl_swizzle_f32_4_rgb(v: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](v[0], v[1], v[2], 0.0)

fn _crossgl_swizzle_f32_4_xyz(v: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](v[0], v[1], v[2], 0.0)


@value
struct MaterialProperties:
    var albedo: SIMD[DType.float32, 4]
    var metallic: Float32
    var roughness: Float32
    var ao: Float32
    var emission: SIMD[DType.float32, 4]
    var normal_scale: Float32
    var height_scale: Float32
    var has_albedo_map: Bool
    var has_normal_map: Bool
    var has_metallic_roughness_map: Bool
    var has_ao_map: Bool
    var has_emission_map: Bool
    var has_height_map: Bool

@value
struct LightData:
    var position: SIMD[DType.float32, 4]
    var direction: SIMD[DType.float32, 4]
    var color: SIMD[DType.float32, 4]
    var intensity: Float32
    var range: Float32
    var inner_cone_angle: Float32
    var outer_cone_angle: Float32
    var type_: Int32
    var cast_shadows: Bool
    var light_view_proj: CrossGLMatrixF32C4R4

@value
struct EnvironmentData:
    var irradiance_map: TextureCube
    var prefilter_map: TextureCube
    var brdf_lut: Texture2D
    var max_reflection_lod: Float32
    var exposure: Float32
    var ambient_color: SIMD[DType.float32, 4]

@value
struct CameraData:
    var position: SIMD[DType.float32, 4]
    var forward: SIMD[DType.float32, 4]
    var up: SIMD[DType.float32, 4]
    var right: SIMD[DType.float32, 4]
    var view_matrix: CrossGLMatrixF32C4R4
    var projection_matrix: CrossGLMatrixF32C4R4
    var view_projection_matrix: CrossGLMatrixF32C4R4
    var near_plane: Float32
    var far_plane: Float32
    var fov: Float32
    var screen_size: SIMD[DType.float32, 2]

@value
struct RenderSettings:
    var enable_ibl: Bool
    var enable_shadows: Bool
    var enable_normal_mapping: Bool
    var enable_parallax_mapping: Bool
    var enable_tone_mapping: Bool
    var enable_gamma_correction: Bool
    var shadow_cascade_count: Int32
    var shadow_bias: Float32
    var max_lights: Int32
    var lod_bias: Float32

alias PI = 3.14159265359
alias EPSILON = 0.0001
alias MAX_LIGHTS = 32
alias MAX_SHADOW_CASCADES = 4

fn getNormalFromMap(normal_map: Texture2D, uv: SIMD[DType.float32, 2], tbn: CrossGLMatrixF32C3R3, scale: Float32) -> SIMD[DType.float32, 4]:
    var tangent_normal: SIMD[DType.float32, 4] = _crossgl_vec3_sub_f32_vs(_crossgl_vec3_mul_f32_vs(_crossgl_swizzle_f32_4_xyz(sample(normal_map, uv)), 2.0), 1.0)
    var __cgl_swizzle_0: SIMD[DType.float32, 2] = scale
    tangent_normal[0] *= __cgl_swizzle_0[0]
    tangent_normal[1] *= __cgl_swizzle_0[1]
    return normalize((tbn * tangent_normal))

fn parallaxMapping(height_map: Texture2D, uv: SIMD[DType.float32, 2], view_dir: SIMD[DType.float32, 4], height_scale: Float32) -> SIMD[DType.float32, 2]:
    var height: Float32 = sample(height_map, uv)[0]
    var p: SIMD[DType.float32, 2] = ((SIMD[DType.float32, 2](view_dir[0], view_dir[1]) / view_dir[2]) * (height * height_scale))
    return (uv - p)

fn distributionGGX(N: SIMD[DType.float32, 4], H: SIMD[DType.float32, 4], roughness: Float32) -> Float32:
    var a: Float32 = (roughness * roughness)
    var a2: Float32 = (a * a)
    var NdotH: Float32 = max(Float32(dot_product(N, H)), Float32(0.0))
    var NdotH2: Float32 = (NdotH * NdotH)
    var num: Float32 = a2
    var denom: Float32 = ((NdotH2 * (a2 - 1.0)) + 1.0)
    denom = ((PI * denom) * denom)
    return (num / max(Float32(denom), Float32(EPSILON)))

fn geometrySchlickGGX(NdotV: Float32, roughness: Float32) -> Float32:
    var r: Float32 = (roughness + 1.0)
    var k: Float32 = ((r * r) / 8.0)
    var num: Float32 = NdotV
    var denom: Float32 = ((NdotV * (1.0 - k)) + k)
    return (num / max(Float32(denom), Float32(EPSILON)))

fn geometrySmith(N: SIMD[DType.float32, 4], V: SIMD[DType.float32, 4], L: SIMD[DType.float32, 4], roughness: Float32) -> Float32:
    var NdotV: Float32 = max(Float32(dot_product(N, V)), Float32(0.0))
    var NdotL: Float32 = max(Float32(dot_product(N, L)), Float32(0.0))
    var ggx2: Float32 = geometrySchlickGGX(NdotV, roughness)
    var ggx1: Float32 = geometrySchlickGGX(NdotL, roughness)
    return (ggx1 * ggx2)

fn fresnelSchlick(cosTheta: Float32, F0: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return _crossgl_vec3_add_f32_vv(F0, _crossgl_vec3_mul_f32_vs(_crossgl_vec3_sub_f32_sv(1.0, F0), power(max(Float32((1.0 - cosTheta)), Float32(0.0)), 5.0)))

fn fresnelSchlickRoughness(cosTheta: Float32, F0: SIMD[DType.float32, 4], roughness: Float32) -> SIMD[DType.float32, 4]:
    return _crossgl_vec3_add_f32_vv(F0, _crossgl_vec3_mul_f32_vs(_crossgl_vec3_sub_f32_vv(max(_crossgl_vec3_splat_f32((1.0 - roughness)), F0), F0), power(max(Float32((1.0 - cosTheta)), Float32(0.0)), 5.0)))

fn calculateShadow(shadow_map: Texture2D, frag_pos_light_space: SIMD[DType.float32, 4], bias: Float32) -> Float32:
    var proj_coords: SIMD[DType.float32, 4] = _crossgl_vec3_div_f32_vs(SIMD[DType.float32, 4](frag_pos_light_space[0], frag_pos_light_space[1], frag_pos_light_space[2], 0.0), frag_pos_light_space[3])
    proj_coords = _crossgl_vec3_add_f32_vs(_crossgl_vec3_mul_f32_vs(proj_coords, 0.5), 0.5)
    if (proj_coords[2] > 1.0):
        return 0.0
    var shadow: Float32 = 0.0
    var texel_size: SIMD[DType.float32, 2] = (1.0 / (texture_size(shadow_map, 0)).cast[DType.float32]())
    var x: Int32 = (-1)
    while (x <= 1):
        var y: Int32 = (-1)
        while (y <= 1):
            var pcf_depth: Float32 = sample(shadow_map, (SIMD[DType.float32, 2](proj_coords[0], proj_coords[1]) + (SIMD[DType.float32, 2]((x).cast[DType.float32](), (y).cast[DType.float32]()) * texel_size)))[0]
            shadow += (1.0 if ((proj_coords[2] - bias) > pcf_depth) else 0.0)
            y += 1
        x += 1
    return (shadow / 9.0)

fn calculateIBL(N: SIMD[DType.float32, 4], V: SIMD[DType.float32, 4], albedo: SIMD[DType.float32, 4], metallic: Float32, roughness: Float32, env: EnvironmentData) -> SIMD[DType.float32, 4]:
    var F0: SIMD[DType.float32, 4] = lerp(SIMD[DType.float32, 4](0.04, 0.04, 0.04, 0.0), albedo, metallic)
    var F: SIMD[DType.float32, 4] = fresnelSchlickRoughness(max(Float32(dot_product(N, V)), Float32(0.0)), F0, roughness)
    var kS: SIMD[DType.float32, 4] = F
    var kD: SIMD[DType.float32, 4] = _crossgl_vec3_sub_f32_sv(1.0, kS)
    kD *= (1.0 - metallic)
    var irradiance: SIMD[DType.float32, 4] = _crossgl_swizzle_f32_4_rgb(sample(env.irradiance_map, N))
    var diffuse: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_vv(irradiance, albedo)
    var R: SIMD[DType.float32, 4] = reflect((-V), N)
    var prefiltered_color: SIMD[DType.float32, 4] = _crossgl_swizzle_f32_4_rgb(sample_lod(env.prefilter_map, R, (roughness * env.max_reflection_lod)))
    var brdf: SIMD[DType.float32, 2] = _crossgl_swizzle_f32_4_rg(sample(env.brdf_lut, SIMD[DType.float32, 2](max(Float32(dot_product(N, V)), Float32(0.0)), roughness)))
    var specular: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_vv(prefiltered_color, _crossgl_vec3_add_f32_vs(_crossgl_vec3_mul_f32_vs(F, brdf[0]), brdf[1]))
    return _crossgl_vec3_mul_f32_vs(_crossgl_vec3_add_f32_vv(_crossgl_vec3_mul_f32_vv(kD, diffuse), specular), env.exposure)

fn calculateDirectLighting(N: SIMD[DType.float32, 4], V: SIMD[DType.float32, 4], L: SIMD[DType.float32, 4], albedo: SIMD[DType.float32, 4], metallic: Float32, roughness: Float32, light_color: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    var H: SIMD[DType.float32, 4] = normalize(_crossgl_vec3_add_f32_vv(V, L))
    var F0: SIMD[DType.float32, 4] = lerp(SIMD[DType.float32, 4](0.04, 0.04, 0.04, 0.0), albedo, metallic)
    var NDF: Float32 = distributionGGX(N, H, roughness)
    var G: Float32 = geometrySmith(N, V, L, roughness)
    var F: SIMD[DType.float32, 4] = fresnelSchlick(max(Float32(dot_product(H, V)), Float32(0.0)), F0)
    var kS: SIMD[DType.float32, 4] = F
    var kD: SIMD[DType.float32, 4] = _crossgl_vec3_sub_f32_vv(SIMD[DType.float32, 4](1.0, 1.0, 1.0, 0.0), kS)
    kD *= (1.0 - metallic)
    var numerator: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_sv((NDF * G), F)
    var denominator: Float32 = (((4.0 * max(Float32(dot_product(N, V)), Float32(0.0))) * max(Float32(dot_product(N, L)), Float32(0.0))) + EPSILON)
    var specular: SIMD[DType.float32, 4] = _crossgl_vec3_div_f32_vs(numerator, denominator)
    var NdotL: Float32 = max(Float32(dot_product(N, L)), Float32(0.0))
    return _crossgl_vec3_mul_f32_vs(_crossgl_vec3_mul_f32_vv(_crossgl_vec3_add_f32_vv(_crossgl_vec3_div_f32_vs(_crossgl_vec3_mul_f32_vv(kD, albedo), PI), specular), light_color), NdotL)

fn calculateAttenuation(light: LightData, frag_pos: SIMD[DType.float32, 4]) -> Float32:
    if (light.type_ == 0):
        return 1.0
    var distance: Float32 = magnitude(_crossgl_vec3_sub_f32_vv(light.position, frag_pos))
    if (light.type_ == 1):
        var attenuation: Float32 = (1.0 / (distance * distance))
        return (attenuation * smoothstep(Float32(light.range), Float32(0.0), Float32(distance)))
    if (light.type_ == 2):
        var light_dir: SIMD[DType.float32, 4] = normalize(_crossgl_vec3_sub_f32_vv(light.position, frag_pos))
        var theta: Float32 = dot_product(light_dir, normalize((-light.direction)))
        var epsilon: Float32 = (light.inner_cone_angle - light.outer_cone_angle)
        var intensity: Float32 = clamp(((theta - light.outer_cone_angle) / epsilon), 0.0, 1.0)
        var attenuation: Float32 = (1.0 / (distance * distance))
        return ((attenuation * intensity) * smoothstep(Float32(light.range), Float32(0.0), Float32(distance)))
    return 0.0

fn reinhardToneMapping(color: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return _crossgl_vec3_div_f32_vv(color, _crossgl_vec3_add_f32_vv(color, SIMD[DType.float32, 4](1.0, 1.0, 1.0, 0.0)))

fn acesToneMapping(color: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    var a: Float32 = 2.51
    var b: Float32 = 0.03
    var c: Float32 = 2.43
    var d: Float32 = 0.59
    var e: Float32 = 0.14
    return clamp(_crossgl_vec3_div_f32_vv(_crossgl_vec3_mul_f32_vv(color, _crossgl_vec3_add_f32_vs(_crossgl_vec3_mul_f32_sv(a, color), b)), _crossgl_vec3_add_f32_vs(_crossgl_vec3_mul_f32_vv(color, _crossgl_vec3_add_f32_vs(_crossgl_vec3_mul_f32_sv(c, color), d)), e)), Float32(0.0), Float32(1.0))

fn gammaCorrection(color: SIMD[DType.float32, 4], gamma: Float32) -> SIMD[DType.float32, 4]:
    return power(color, _crossgl_vec3_splat_f32((1.0 / gamma)))

# Vertex Shader
@value
struct VertexInput:
    var position: SIMD[DType.float32, 4]
    var normal: SIMD[DType.float32, 4]
    var tangent: SIMD[DType.float32, 4]
    var uv: SIMD[DType.float32, 2]
    var color: SIMD[DType.float32, 4]

@value
struct VertexOutput:
    var clip_position: SIMD[DType.float32, 4]
    var world_position: SIMD[DType.float32, 4]
    var world_normal: SIMD[DType.float32, 4]
    var world_tangent: SIMD[DType.float32, 4]
    var world_bitangent: SIMD[DType.float32, 4]
    var uv: SIMD[DType.float32, 2]
    var color: SIMD[DType.float32, 4]
    var tbn_matrix: CrossGLMatrixF32C3R3
    var shadow_coords: InlineArray[SIMD[DType.float32, 4], MAX_SHADOW_CASCADES]

var model_matrix: CrossGLMatrixF32C4R4 = CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
var view_matrix: CrossGLMatrixF32C4R4 = CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
var projection_matrix: CrossGLMatrixF32C4R4 = CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
var normal_matrix: CrossGLMatrixF32C3R3 = CrossGLMatrixF32C3R3(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
var camera = CameraData(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)), CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)), CrossGLMatrixF32C4R4(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)), 0.0, 0.0, 0.0, SIMD[DType.float32, 2](0.0, 0.0))
var settings = RenderSettings(False, False, False, False, False, False, 0, 0.0, 0, 0.0)
var shadow_matrices = InlineArray[CrossGLMatrixF32C4R4, MAX_SHADOW_CASCADES](unsafe_uninitialized=True)
# CrossGL shader stage: vertex
fn vertex_main(input: VertexInput) -> VertexOutput:
    var output = VertexOutput(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 2](0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), CrossGLMatrixF32C3R3(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)), InlineArray[SIMD[DType.float32, 4], MAX_SHADOW_CASCADES](unsafe_uninitialized=True))
    var world_pos: SIMD[DType.float32, 4] = (model_matrix * SIMD[DType.float32, 4](input.position[0], input.position[1], input.position[2], 1.0))
    output.world_position = SIMD[DType.float32, 4](world_pos[0], world_pos[1], world_pos[2], 0.0)
    output.clip_position = (camera.view_projection_matrix * world_pos)
    output.world_normal = normalize((normal_matrix * input.normal))
    output.world_tangent = normalize((normal_matrix * input.tangent))
    output.world_bitangent = cross_product(output.world_normal, output.world_tangent)
    output.tbn_matrix = CrossGLMatrixF32C3R3(SIMD[DType.float32, 4](output.world_tangent[0], output.world_tangent[1], output.world_tangent[2], 0.0), SIMD[DType.float32, 4](output.world_bitangent[0], output.world_bitangent[1], output.world_bitangent[2], 0.0), SIMD[DType.float32, 4](output.world_normal[0], output.world_normal[1], output.world_normal[2], 0.0))
    output.uv = input.uv
    output.color = input.color
    if settings.enable_shadows:
        var i: Int32 = 0
        while ((i < settings.shadow_cascade_count) and (i < MAX_SHADOW_CASCADES)):
            output.shadow_coords[int(i)] = (shadow_matrices[int(i)] * world_pos)
            i += 1
    return output

# Fragment Shader
@value
struct FragmentInput:
    var world_position: SIMD[DType.float32, 4]
    var world_normal: SIMD[DType.float32, 4]
    var world_tangent: SIMD[DType.float32, 4]
    var world_bitangent: SIMD[DType.float32, 4]
    var uv: SIMD[DType.float32, 2]
    var color: SIMD[DType.float32, 4]
    var tbn_matrix: CrossGLMatrixF32C3R3
    var shadow_coords: InlineArray[SIMD[DType.float32, 4], MAX_SHADOW_CASCADES]

# CrossGL resource metadata: name=albedo_map kind=texture set=0 binding=0 binding_source=automatic
var albedo_map: Texture2D = Texture2D()
# CrossGL resource metadata: name=normal_map kind=texture set=0 binding=1 binding_source=automatic
var normal_map: Texture2D = Texture2D()
# CrossGL resource metadata: name=metallic_roughness_map kind=texture set=0 binding=2 binding_source=automatic
var metallic_roughness_map: Texture2D = Texture2D()
# CrossGL resource metadata: name=ao_map kind=texture set=0 binding=3 binding_source=automatic
var ao_map: Texture2D = Texture2D()
# CrossGL resource metadata: name=emission_map kind=texture set=0 binding=4 binding_source=automatic
var emission_map: Texture2D = Texture2D()
# CrossGL resource metadata: name=height_map kind=texture set=0 binding=5 binding_source=automatic
var height_map: Texture2D = Texture2D()
# CrossGL resource metadata: name=shadow_maps kind=texture set=0 binding=6 binding_source=automatic count=4
var shadow_maps = InlineArray[Texture2D, MAX_SHADOW_CASCADES](unsafe_uninitialized=True)
var material = MaterialProperties(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0.0, SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0), 0.0, 0.0, False, False, False, False, False, False)
var environment = EnvironmentData(TextureCube(), TextureCube(), Texture2D(), 0.0, 0.0, SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
var lights = InlineArray[LightData, MAX_LIGHTS](unsafe_uninitialized=True)
var active_light_count: Int32 = 0
# CrossGL shader stage: fragment
fn fragment_main(input: FragmentInput) -> SIMD[DType.float32, 4]:
    var uv: SIMD[DType.float32, 2] = input.uv
    if (settings.enable_parallax_mapping and material.has_height_map):
        var view_dir: SIMD[DType.float32, 4] = normalize(_crossgl_vec3_sub_f32_vv(camera.position, input.world_position))
        var tangent_view_dir: SIMD[DType.float32, 4] = (transpose(input.tbn_matrix) * view_dir)
        uv = parallaxMapping(height_map, uv, tangent_view_dir, material.height_scale)
    var albedo: SIMD[DType.float32, 4] = material.albedo
    if material.has_albedo_map:
        albedo *= _crossgl_swizzle_f32_4_rgb(sample(albedo_map, uv))
    albedo *= SIMD[DType.float32, 4](input.color[0], input.color[1], input.color[2], 0.0)
    var metallic: Float32 = material.metallic
    var roughness: Float32 = material.roughness
    if material.has_metallic_roughness_map:
        var mr_sample: SIMD[DType.float32, 4] = _crossgl_swizzle_f32_4_rgb(sample(metallic_roughness_map, uv))
        metallic *= mr_sample[2]
        roughness *= mr_sample[1]
    var ao: Float32 = material.ao
    if material.has_ao_map:
        ao *= sample(ao_map, uv)[0]
    var emission: SIMD[DType.float32, 4] = material.emission
    if material.has_emission_map:
        emission *= _crossgl_swizzle_f32_4_rgb(sample(emission_map, uv))
    var N: SIMD[DType.float32, 4] = normalize(input.world_normal)
    if (settings.enable_normal_mapping and material.has_normal_map):
        N = getNormalFromMap(normal_map, uv, input.tbn_matrix, material.normal_scale)
    var V: SIMD[DType.float32, 4] = normalize(_crossgl_vec3_sub_f32_vv(camera.position, input.world_position))
    var Lo: SIMD[DType.float32, 4] = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
    var i: Int32 = 0
    while ((i < active_light_count) and (i < MAX_LIGHTS)):
        var light: LightData = lights[int(i)]
        var L: SIMD[DType.float32, 4]
        if (light.type_ == 0):
            L = normalize((-light.direction))
        else:
            L = normalize(_crossgl_vec3_sub_f32_vv(light.position, input.world_position))
        var attenuation: Float32 = calculateAttenuation(light, input.world_position)
        var radiance: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_vs(_crossgl_vec3_mul_f32_vs(light.color, light.intensity), attenuation)
        var shadow: Float32 = 0.0
        if ((settings.enable_shadows and light.cast_shadows) and (i < settings.shadow_cascade_count)):
            shadow = calculateShadow(shadow_maps[int(i)], input.shadow_coords[int(i)], settings.shadow_bias)
        Lo += _crossgl_vec3_mul_f32_vs(calculateDirectLighting(N, V, L, albedo, metallic, roughness, radiance), (1.0 - shadow))
        i += 1
    var ambient: SIMD[DType.float32, 4] = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
    if settings.enable_ibl:
        ambient = calculateIBL(N, V, albedo, metallic, roughness, environment)
    else:
        ambient = _crossgl_vec3_mul_f32_vs(_crossgl_vec3_mul_f32_vv(environment.ambient_color, albedo), ao)
    var color: SIMD[DType.float32, 4] = _crossgl_vec3_add_f32_vv(_crossgl_vec3_add_f32_vv(ambient, Lo), emission)
    if settings.enable_tone_mapping:
        color = acesToneMapping(color)
    if settings.enable_gamma_correction:
        color = gammaCorrection(color, 2.2)
    return SIMD[DType.float32, 4](color[0], color[1], color[2], input.color[3])

# Compute Shader
# CrossGL resource metadata: name=environment_map kind=texture set=0 binding=10 binding_source=automatic
var environment_map: TextureCube = TextureCube()
# CrossGL resource metadata: name=irradiance_map kind=image set=0 binding=0 binding_source=automatic
var irradiance_map: ImageCubeFloat4 = ImageCubeFloat4()
var face_index: Int32 = 0
var mip_level: Int32 = 0
fn getSamplingVector(uv: SIMD[DType.float32, 2], face: Int32) -> SIMD[DType.float32, 4]:
    var result: SIMD[DType.float32, 4]
    if face == 0:
        result = SIMD[DType.float32, 4](1.0, (-uv[1]), (-uv[0]), 0.0)
    elif face == 1:
        result = SIMD[DType.float32, 4]((-1.0), (-uv[1]), uv[0], 0.0)
    elif face == 2:
        result = SIMD[DType.float32, 4](uv[0], 1.0, uv[1], 0.0)
    elif face == 3:
        result = SIMD[DType.float32, 4](uv[0], (-1.0), (-uv[1]), 0.0)
    elif face == 4:
        result = SIMD[DType.float32, 4](uv[0], (-uv[1]), 1.0, 0.0)
    elif face == 5:
        result = SIMD[DType.float32, 4]((-uv[0]), (-uv[1]), (-1.0), 0.0)
    return normalize(result)

# CrossGL shader stage: compute
fn precompute_environment() -> None:
    var coord: SIMD[DType.int32, 2] = SIMD[DType.int32, 2](SIMD[DType.uint32, 4](global_idx_uint.x, global_idx_uint.y, global_idx_uint.z, 0)[0].cast[DType.int32](), SIMD[DType.uint32, 4](global_idx_uint.x, global_idx_uint.y, global_idx_uint.z, 0)[1].cast[DType.int32]())
    var size: SIMD[DType.int32, 2] = image_size(irradiance_map)
    if ((coord[0] >= size[0]) or (coord[1] >= size[1])):
        return None
    var uv: SIMD[DType.float32, 2] = ((SIMD[DType.float32, 2](coord[0].cast[DType.float32](), coord[1].cast[DType.float32]()) + 0.5) / SIMD[DType.float32, 2](size[0].cast[DType.float32](), size[1].cast[DType.float32]()))
    uv = ((uv * 2.0) - 1.0)
    var N: SIMD[DType.float32, 4] = getSamplingVector(uv, face_index)
    var irradiance: SIMD[DType.float32, 4] = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
    var sample_count: Float32 = 0.0
    var phi: Float32 = 0.0
    while (phi < (2.0 * PI)):
        var theta: Float32 = 0.0
        while (theta < (0.5 * PI)):
            var tangent_sample: SIMD[DType.float32, 4] = SIMD[DType.float32, 4]((sin(theta) * cos(phi)), (sin(theta) * sin(phi)), cos(theta), 0.0)
            var up: SIMD[DType.float32, 4] = (SIMD[DType.float32, 4](0.0, 0.0, 1.0, 0.0) if (abs(N[2]) < 0.999) else SIMD[DType.float32, 4](1.0, 0.0, 0.0, 0.0))
            var right: SIMD[DType.float32, 4] = normalize(cross_product(up, N))
            up = normalize(cross_product(N, right))
            var sample_vec: SIMD[DType.float32, 4] = _crossgl_vec3_add_f32_vv(_crossgl_vec3_add_f32_vv(_crossgl_vec3_mul_f32_sv(tangent_sample[0], right), _crossgl_vec3_mul_f32_sv(tangent_sample[1], up)), _crossgl_vec3_mul_f32_sv(tangent_sample[2], N))
            irradiance += _crossgl_vec3_mul_f32_vs(_crossgl_vec3_mul_f32_vs(_crossgl_swizzle_f32_4_rgb(sample(environment_map, sample_vec)), cos(theta)), sin(theta))
            sample_count += 1
            theta += 0.025
        phi += 0.025
    irradiance = _crossgl_vec3_div_f32_vs(_crossgl_vec3_mul_f32_sv(PI, irradiance), sample_count)
    image_store(irradiance_map, SIMD[DType.int32, 4](coord[0], coord[1], face_index, 0), SIMD[DType.float32, 4](irradiance[0], irradiance[1], irradiance[2], 1.0))

