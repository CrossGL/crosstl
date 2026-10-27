"""Aggregate-inout-parameter, out-parameter-forwarding, vector-unary-operators, vector-scalar-arithmetic,
matrix-array and vector-conversion probes: their C-family form (:data:`FAMILY`) and the languages that write them
differently.

The matrices of the matrix-array probe are symmetric and it writes only their diagonal, so the probe reads the same
whether a language fills and indexes a matrix by columns, as GLSL does, or by rows, as HLSL does. Metal passes
references as ``thread`` references, CUDA and HIP as references, and OpenCL C, WGSL and Rust as pointers or mutable
borrows. CUDA and HIP vector types have no operators or conversions, and CUDA, HIP, OpenCL C and Mojo have no
matrices. OpenCL C has no bool vectors, so it converts the mask its comparison gives.
"""

from __future__ import annotations

from .base import ABSENT, PAIR, Template
from .statements import c_text

# absorb changes a member of one argument and two elements of the other, after reading what the caller stored.
ABSORB = (
    "void absorb(inout Pair q, inout float v[3]) {\n    q.a = q.a + v[0];\n    v[1] = q.b * 2.0;\n    q.b = 3.0;\n"
    "    v[2] = v[2] + q.a;\n}"
)
# outer hands its own out and inout parameters on to inner, then changes them again.
FORWARD = (
    "void inner(float x, out float r, inout float s) {\n    r = x * 2.0;\n    s = s + r;\n}\n\n"
    "void outer(float x, out float r, inout float s) {\n    inner(x + 1.0, r, s);\n    r = r + 0.5;\n"
    "    s = s * 2.0;\n}"
)
_AGGREGATE_SUM = (
    "p.a + p.b * 16.0 + values[0] * 256.0 + values[1] * 4096.0 + values[2] * 65536.0"
)
_UNARY_SUM = (
    "n{x} + n{y} * 16.0 + {f}(b{x}) * 256.0 + {f}(b{y}) * 4096.0 + {f}(c{x}) * 65536.0"
    " + {f}(c{y}) * 1048576.0"
)
_SCALAR_SUM = (
    "a{x} + c{x} * 2.0 + (a{y} + b{x} * 8.0 + c{y} * 256.0) * 4.0 + {f}(e{x} + g{x} * 32 + e{y} - g{y}) * 16384.0"
    " + b{y} * 0.5"
)
_MATRIX_SUM = (
    "ms[0]{c0}{r0} + ms[0]{c0}{r1} * 32.0 + ms[0]{c1}{r1} * 128.0 + ms[1]{c0}{r0} * 1024.0"
    " + ms[1]{c1}{r0} * 8192.0 + ms[1]{c1}{r1} * 131072.0"
)
_CONVERSION_SUM = (
    "f{x} + f{y} * 16.0 + {f}(t{x}) * 256.0 + {f}(t{y}) * 4096.0 + {f}(h{y}) * 65536.0"
    " + (w{x} + w{y} * 2.0) * 1048576.0"
)
_C = {"x": ".x", "y": ".y", "f": "float"}
_INDEXED = {"x": "[0]", "y": "[1]", "f": "Float32"}

FAMILY = {
    # A struct and an array passed whole to inout parameters: the caller sees what the callee wrote to both.
    "aggregate-inout-parameter": Template(
        "Pair p;\np.a = fin[i];\np.b = float(iin[i]);\nfloat values[3];\nvalues[0] = 1.0;\nvalues[1] = 2.0;\n"
        f"values[2] = 4.0;\nabsorb(p, values);\noutp[i] = {_AGGREGATE_SUM};",
        PAIR + "\n" + ABSORB,
    ),
    "out-parameter-forwarding": Template(
        "float a;\nfloat b = float(iin[i]);\nouter(fin[i], a, b);\noutp[i] = a + b * 16.0;",
        FORWARD,
    ),
    "vector-unary-operators": Template(
        "$vec2 v = $vec2(fin[i], -2.0);\n$vec2 n = -v;\n$ivec2 a = $ivec2(iin[i], -3);\n$ivec2 b = -a;\n"
        f"$ivec2 c = ~a;\noutp[i] = {_UNARY_SUM.format(**_C)};"
    ),
    # A scalar on either side of a vector operand; s is a power of two, so every quotient is exact.
    "vector-scalar-arithmetic": Template(
        "float s = float(1 << (iin[i] & 3));\n$vec2 v = $vec2(fin[i], s);\n$vec2 a = 1.0 - v;\n"
        "$vec2 b = 16.0 / $vec2(s, 2.0);\n$vec2 c = v + 3.0;\n$ivec2 k = $ivec2(iin[i], 7);\n$ivec2 e = 20 - k;\n"
        f"$ivec2 g = 1 + k;\noutp[i] = {_SCALAR_SUM.format(**_C)};"
    ),
    # The input picks which element of the array is scaled and which has its diagonal changed.
    "matrix-array": Template(
        "float f = float(iin[i]);\n$mat2 ms[2];\nms[0] = $mat2(f, 1.0, 1.0, 3.0);\nms[1] = $mat2(2.0, 5.0, 5.0, f);\n"
        "int j = iin[i] & 1;\nms[j] = ms[j] * 2.0;\nms[1 - j][1][1] = 4.0;\noutp[i] = "
        + _MATRIX_SUM.format(c0="[0]", c1="[1]", r0="[0]", r1="[1]")
        + ";"
    ),
    # Signed to float, float to signed (toward zero), signed to unsigned (modulo 2^32), and through bool vectors.
    "vector-conversion": Template(
        "$ivec2 a = $ivec2(iin[i], -5);\n$vec2 f = $vec2(a);\n$vec2 g = $vec2(fin[i] * 4.0, -2.75);\n"
        "$ivec2 t = $ivec2(g);\n$uvec2 u = $uvec2(a);\n$uvec2 h = u >> 28u;\n$bvec2 z = $bvec2(a);\n"
        f"$vec2 w = $vec2(z);\noutp[i] = {_CONVERSION_SUM.format(**_C)};"
    ),
}

METAL = {
    "aggregate-inout-parameter": Template(
        FAMILY["aggregate-inout-parameter"].body,
        PAIR
        + "\n"
        + ABSORB.replace("inout Pair q", "thread Pair& q").replace(
            "inout float v[3]", "thread float (&v)[3]"
        ),
    ),
    "out-parameter-forwarding": Template(
        FAMILY["out-parameter-forwarding"].body,
        FORWARD.replace("inout float", "thread float&").replace(
            "out float", "thread float&"
        ),
    ),
}

CUDA = {
    "aggregate-inout-parameter": Template(
        c_text(FAMILY["aggregate-inout-parameter"].body, "unsigned int"),
        PAIR
        + "\n__device__ "
        + c_text(ABSORB, "unsigned int")
        .replace("inout Pair q", "Pair& q")
        .replace("inout float v[3]", "float (&v)[3]"),
    ),
    "out-parameter-forwarding": Template(
        c_text(FAMILY["out-parameter-forwarding"].body, "unsigned int"),
        "__device__ "
        + c_text(FORWARD, "unsigned int")
        .replace("inout float", "float&")
        .replace("out float", "float&")
        .replace("void outer", "__device__ void outer"),
    ),
    "vector-unary-operators": ABSENT,
    "vector-scalar-arithmetic": ABSENT,
    "matrix-array": ABSENT,
    "vector-conversion": ABSENT,
}

OPENCL = {
    "aggregate-inout-parameter": Template(
        c_text(FAMILY["aggregate-inout-parameter"].body, "uint").replace(
            "absorb(p,", "absorb(&p,"
        ),
        "typedef struct {\n    float a;\n    float b;\n} Pair;\n\n"
        "void absorb(Pair* q, float v[3]) {\n    q->a = q->a + v[0];\n    v[1] = q->b * 2.0f;\n    q->b = 3.0f;\n"
        "    v[2] = v[2] + q->a;\n}",
    ),
    "out-parameter-forwarding": Template(
        c_text(FAMILY["out-parameter-forwarding"].body, "uint").replace(
            "a, b);", "&a, &b);"
        ),
        "void inner(float x, float* r, float* s) {\n    *r = x * 2.0f;\n    *s = *s + *r;\n}\n\n"
        "void outer(float x, float* r, float* s) {\n    inner(x + 1.0f, r, s);\n    *r = *r + 0.5f;\n"
        "    *s = *s * 2.0f;\n}",
    ),
    "vector-unary-operators": Template(
        "float2 v = (float2)(fin[i], -2.0f);\nfloat2 n = -v;\nint2 a = (int2)(iin[i], -3);\nint2 b = -a;\n"
        "int2 c = ~a;\noutp[i] = " + c_text(_UNARY_SUM.format(**_C), "uint") + ";"
    ),
    "vector-scalar-arithmetic": Template(
        "float s = (float)(1 << (iin[i] & 3));\nfloat2 v = (float2)(fin[i], s);\nfloat2 a = 1.0f - v;\n"
        "float2 b = 16.0f / (float2)(s, 2.0f);\nfloat2 c = v + 3.0f;\nint2 k = (int2)(iin[i], 7);\n"
        "int2 e = 20 - k;\nint2 g = 1 + k;\noutp[i] = "
        + c_text(_SCALAR_SUM.format(**_C), "uint")
        + ";"
    ),
    "matrix-array": ABSENT,
    "vector-conversion": Template(
        "int2 a = (int2)(iin[i], -5);\nfloat2 f = convert_float2(a);\nfloat2 g = (float2)(fin[i] * 4.0f, -2.75f);\n"
        "int2 t = convert_int2(g);\nuint2 u = convert_uint2(a);\nuint2 h = u >> 28u;\n"
        "int2 z = -(a != (int2)(0, 0));\nfloat2 w = convert_float2(z);\noutp[i] = "
        + c_text(_CONVERSION_SUM.format(**_C), "uint")
        + ";"
    ),
}

WGSL = {
    "aggregate-inout-parameter": Template(
        "var p: Pair;\np.a = fin[i];\np.b = f32(iin[i]);\nvar values = array<f32, 3>(1.0, 2.0, 4.0);\n"
        f"absorb(&p, &values);\noutp[i] = {_AGGREGATE_SUM};",
        "struct Pair {\n    a: f32,\n    b: f32,\n}\n\n"
        "fn absorb(q: ptr<function, Pair>, v: ptr<function, array<f32, 3>>) {\n    (*q).a = (*q).a + (*v)[0];\n"
        "    (*v)[1] = (*q).b * 2.0;\n    (*q).b = 3.0;\n    (*v)[2] = (*v)[2] + (*q).a;\n}",
    ),
    "out-parameter-forwarding": Template(
        "var a: f32;\nvar b = f32(iin[i]);\nouter(fin[i], &a, &b);\noutp[i] = a + b * 16.0;",
        "fn inner(x: f32, r: ptr<function, f32>, s: ptr<function, f32>) {\n    *r = x * 2.0;\n    *s = *s + *r;\n}\n\n"
        "fn outer(x: f32, r: ptr<function, f32>, s: ptr<function, f32>) {\n    inner(x + 1.0, r, s);\n"
        "    *r = *r + 0.5;\n    *s = *s * 2.0;\n}",
    ),
    "vector-unary-operators": Template(
        "let v = vec2<f32>(fin[i], -2.0);\nlet n = -v;\nlet a = vec2<i32>(iin[i], -3);\nlet b = -a;\nlet c = ~a;\n"
        f"outp[i] = {_UNARY_SUM.format(x='.x', y='.y', f='f32')};"
    ),
    "vector-scalar-arithmetic": Template(
        "let s = f32(1 << u32(iin[i] & 3));\nlet v = vec2<f32>(fin[i], s);\nlet a = 1.0 - v;\n"
        "let b = 16.0 / vec2<f32>(s, 2.0);\nlet c = v + 3.0;\nlet k = vec2<i32>(iin[i], 7);\nlet e = 20 - k;\n"
        f"let g = 1 + k;\noutp[i] = {_SCALAR_SUM.format(x='.x', y='.y', f='f32')};"
    ),
    "matrix-array": Template(
        "let f = f32(iin[i]);\nvar ms: array<mat2x2<f32>, 2>;\nms[0] = mat2x2<f32>(f, 1.0, 1.0, 3.0);\n"
        "ms[1] = mat2x2<f32>(2.0, 5.0, 5.0, f);\nlet j = iin[i] & 1;\nms[j] = ms[j] * 2.0;\nms[1 - j][1][1] = 4.0;\n"
        "outp[i] = " + _MATRIX_SUM.format(c0="[0]", c1="[1]", r0="[0]", r1="[1]") + ";"
    ),
    "vector-conversion": Template(
        "let a = vec2<i32>(iin[i], -5);\nlet f = vec2<f32>(a);\nlet g = vec2<f32>(fin[i] * 4.0, -2.75);\n"
        "let t = vec2<i32>(g);\nlet u = vec2<u32>(a);\nlet h = u >> vec2<u32>(28u);\nlet z = vec2<bool>(a);\n"
        f"let w = vec2<f32>(z);\noutp[i] = {_CONVERSION_SUM.format(x='.x', y='.y', f='f32')};"
    ),
}

RUST = {
    "aggregate-inout-parameter": Template(
        "let mut p = Pair { a: fin[i], b: iin[i] as f32 };\nlet mut values = [1.0, 2.0, 4.0];\n"
        f"absorb(&mut p, &mut values);\noutp[i] = {_AGGREGATE_SUM};",
        "#[derive(Clone, Copy, Default)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}\n\n"
        "fn absorb(q: &mut Pair, v: &mut [f32; 3]) {\n    q.a = q.a + v[0];\n    v[1] = q.b * 2.0;\n    q.b = 3.0;\n"
        "    v[2] = v[2] + q.a;\n}",
    ),
    "out-parameter-forwarding": Template(
        "let mut a = 0.0;\nlet mut b = iin[i] as f32;\nouter(fin[i], &mut a, &mut b);\noutp[i] = a + b * 16.0;",
        "fn inner(x: f32, r: &mut f32, s: &mut f32) {\n    *r = x * 2.0;\n    *s = *s + *r;\n}\n\n"
        "fn outer(x: f32, r: &mut f32, s: &mut f32) {\n    inner(x + 1.0, r, s);\n    *r = *r + 0.5;\n"
        "    *s = *s * 2.0;\n}",
    ),
    "vector-unary-operators": Template(
        "let v = $vec2::new(fin[i], -2.0);\nlet n = -v;\nlet a = $ivec2::new(iin[i], -3);\nlet b = -a;\nlet c = !a;\n"
        "outp[i] = n.x + n.y * 16.0 + b.x as f32 * 256.0 + b.y as f32 * 4096.0 + c.x as f32 * 65536.0"
        " + c.y as f32 * 1048576.0;"
    ),
    "vector-scalar-arithmetic": Template(
        "let s = (1 << (iin[i] & 3)) as f32;\nlet v = $vec2::new(fin[i], s);\nlet a = 1.0 - v;\n"
        "let b = 16.0 / $vec2::new(s, 2.0);\nlet c = v + 3.0;\nlet k = $ivec2::new(iin[i], 7);\nlet e = 20 - k;\n"
        "let g = 1 + k;\noutp[i] = a.x + c.x * 2.0 + (a.y + b.x * 8.0 + c.y * 256.0) * 4.0"
        " + (e.x + g.x * 32 + e.y - g.y) as f32 * 16384.0 + b.y * 0.5;"
    ),
    "matrix-array": Template(
        "let f = iin[i] as f32;\nlet mut ms = [$mat2::ZERO; 2];\n"
        "ms[0] = $mat2::from_cols($vec2::new(f, 1.0), $vec2::new(1.0, 3.0));\n"
        "ms[1] = $mat2::from_cols($vec2::new(2.0, 5.0), $vec2::new(5.0, f));\nlet j = (iin[i] & 1) as usize;\n"
        "ms[j] = ms[j] * 2.0;\nms[1 - j].y_axis.y = 4.0;\noutp[i] = "
        + _MATRIX_SUM.format(c0=".x_axis", c1=".y_axis", r0=".x", r1=".y")
        + ";"
    ),
    "vector-conversion": Template(
        "let a = $ivec2::new(iin[i], -5);\nlet f = a.as_vec2();\nlet g = $vec2::new(fin[i] * 4.0, -2.75);\n"
        "let t = g.as_ivec2();\nlet u = a.as_uvec2();\nlet h = u >> 28u32;\nlet z = a.cmpne($ivec2::ZERO);\n"
        "let w = $vec2::select(z, $vec2::ONE, $vec2::ZERO);\n"
        "outp[i] = f.x + f.y * 16.0 + t.x as f32 * 256.0 + t.y as f32 * 4096.0 + h.y as f32 * 65536.0"
        " + (w.x + w.y * 2.0) * 1048576.0;"
    ),
}

MOJO = {
    "aggregate-inout-parameter": Template(
        "var p = Pair(fin[i], Float32(iin[i]))\nvar values = Array[Float32, 3](fill=0.0)\nvalues[0] = 1.0\n"
        f"values[1] = 2.0\nvalues[2] = 4.0\nabsorb(p, values)\noutp[i] = {_AGGREGATE_SUM}",
        "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32\n\n\n"
        "def absorb(mut q: Pair, mut v: Array[Float32, 3]):\n    q.a = q.a + v[0]\n    v[1] = q.b * 2.0\n"
        "    q.b = 3.0\n    v[2] = v[2] + q.a",
    ),
    "out-parameter-forwarding": Template(
        "var a: Float32 = 0.0\nvar b = Float32(iin[i])\nouter(fin[i], a, b)\noutp[i] = a + b * 16.0",
        "def inner(x: Float32, mut r: Float32, mut s: Float32):\n    r = x * 2.0\n    s = s + r\n\n\n"
        "def outer(x: Float32, mut r: Float32, mut s: Float32):\n    inner(x + 1.0, r, s)\n    r = r + 0.5\n"
        "    s = s * 2.0",
    ),
    "vector-unary-operators": Template(
        "var v = $vec2(fin[i], -2.0)\nvar n = -v\nvar a = $ivec2(iin[i], -3)\nvar b = -a\nvar c = ~a\n"
        "outp[i] = " + _UNARY_SUM.format(**_INDEXED)
    ),
    "vector-scalar-arithmetic": Template(
        "var s = Float32(1 << (iin[i] & 3))\nvar v = $vec2(fin[i], s)\nvar a = 1.0 - v\n"
        "var b = 16.0 / $vec2(s, 2.0)\nvar c = v + 3.0\nvar k = $ivec2(iin[i], 7)\nvar e = 20 - k\nvar g = 1 + k\n"
        "outp[i] = " + _SCALAR_SUM.format(**_INDEXED)
    ),
    "matrix-array": ABSENT,
    "vector-conversion": Template(
        "var a = $ivec2(iin[i], -5)\nvar f = a.cast[DType.float32]()\nvar g = $vec2(fin[i] * 4.0, -2.75)\n"
        "var t = g.cast[DType.int32]()\nvar u = a.cast[DType.uint32]()\nvar h = u >> 28\n"
        "var z = a.cast[DType.bool]()\nvar w = z.cast[DType.float32]()\n"
        "outp[i] = " + _CONVERSION_SUM.format(**_INDEXED)
    ),
}
