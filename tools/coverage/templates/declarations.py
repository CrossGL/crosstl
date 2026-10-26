"""Vector-conditional, loop-return, wraparound, literal, array-member and multi-declarator probes in the languages
that do not write them as the C family does.

CUDA and HIP build vectors with ``make_float2``, mark device functions ``__device__`` and write ``unsigned int``
for ``uint``; OpenCL C converts with casts and builds vectors with vector literals. WGSL picks a vector with
``select`` and declares one variable per statement (WGSL, section 7). Rust wraps unsigned arithmetic only when
asked to, accepts no ``0X`` prefix or bare ``.5``, and declares several variables at once with a tuple pattern.
"""

from __future__ import annotations

from .base import ABSENT, C_FAMILY, FIND, Template

_C_FIND = FIND.replace("float(k)", "(float)k").replace("-1.0", "-1.0f")
_C_BINS = "typedef struct {\n    float v[3];\n    int n;\n} Bins;"


def _c_casts(name: str) -> str:
    """The C family's probe with casts in place of the functional conversions of ints."""
    return (
        C_FAMILY[name]
        .body.replace("float(b)", "(float)b")
        .replace("float(c >> 28u)", "(float)(c >> 28u)")
        .replace("float(d >> 28u)", "(float)(d >> 28u)")
        .replace("uint(iin[i])", "(uint)iin[i]")
        .replace("uint(iin[i] * 37)", "(uint)(iin[i] * 37)")
        .replace("float(h)", "(float)h")
        .replace("float(m)", "(float)m")
        .replace("float(k + 1)", "(float)(k + 1)")
        .replace("float(b.n)", "(float)b.n")
        .replace("float(j + k)", "(float)(j + k)")
    )


CUDA = {
    "vector-conditional": Template(
        "float2 a = make_float2(fin[i], 1.0f);\nfloat2 b = make_float2(2.0f, fin[i] * 4.0f);\n"
        "float2 r = iin[i] > 7 ? a : b;\noutp[i] = r.x * 16.0f + r.y;"
    ),
    "loop-return": Template(C_FAMILY["loop-return"].body, f"__device__ {FIND}"),
    "unsigned-wraparound": Template(
        _c_casts("unsigned-wraparound").replace("uint", "unsigned int")
    ),
    "numeric-literals": Template(
        _c_casts("numeric-literals").replace("uint", "unsigned int")
    ),
}

OPENCL = {
    "vector-conditional": Template(
        "float2 a = (float2)(fin[i], 1.0f);\nfloat2 b = (float2)(2.0f, fin[i] * 4.0f);\n"
        "float2 r = iin[i] > 7 ? a : b;\noutp[i] = r.x * 16.0f + r.y;"
    ),
    "loop-return": Template(C_FAMILY["loop-return"].body, _C_FIND),
    "unsigned-wraparound": Template(_c_casts("unsigned-wraparound")),
    "numeric-literals": Template(_c_casts("numeric-literals")),
    "struct-array-member": Template(_c_casts("struct-array-member"), _C_BINS),
    "multiple-declarators": Template(_c_casts("multiple-declarators")),
}

WGSL = {
    "vector-conditional": Template(
        "let a = vec2<f32>(fin[i], 1.0);\nlet b = vec2<f32>(2.0, fin[i] * 4.0);\nlet r = select(b, a, iin[i] > 7);\n"
        "outp[i] = r.x * 16.0 + r.y;"
    ),
    "loop-return": Template(
        C_FAMILY["loop-return"].body,
        "fn find(n: i32, x: f32) -> f32 {\n    for (var k = 0; k < 8; k++) {\n        if (k == n) {\n"
        "            return x + f32(k);\n        }\n    }\n    return -1.0;\n}",
    ),
    "unsigned-wraparound": Template(
        "let a = 4294967295u;\nlet n = u32(iin[i]);\nlet b = a + n + 1u;\nlet c = n * 2654435761u;\n"
        "let d = n - 16u;\noutp[i] = f32(b) + f32(c >> 28u) * 16.0 + f32(d >> 28u) * 256.0;"
    ),
    "numeric-literals": Template(
        "let h = 0x1F + 0X0a;\nlet m = 0xFFu & u32(iin[i] * 37);\nlet e = 2.5e-1 + 1E2 + .5;\n"
        "outp[i] = f32(h) + f32(m) * 64.0 + e * 4.0;"
    ),
    "struct-array-member": Template(
        "var b: Bins;\nfor (var k = 0; k < 3; k++) {\n    b.v[k] = f32(k + 1);\n}\nb.n = iin[i];\n"
        "b.v[iin[i] % 3] = fin[i];\noutp[i] = b.v[0] * 64.0 + b.v[1] * 8.0 + b.v[2] + f32(b.n) * 512.0;",
        "struct Bins {\n    v: array<f32, 3>,\n    n: i32,\n}",
    ),
    "multiple-declarators": ABSENT,
}

RUST = {
    "vector-conditional": Template(
        "let a = $vec2::new(fin[i], 1.0);\nlet b = $vec2::new(2.0, fin[i] * 4.0);\n"
        "let r = if iin[i] > 7 { a } else { b };\noutp[i] = r.x * 16.0 + r.y;"
    ),
    "loop-return": Template(
        C_FAMILY["loop-return"].body,
        "fn find(n: i32, x: f32) -> f32 {\n    for k in 0..8 {\n        if k == n {\n"
        "            return x + k as f32;\n        }\n    }\n    -1.0\n}",
    ),
    "unsigned-wraparound": Template(
        "let a: u32 = 4294967295;\nlet n = iin[i] as u32;\nlet b = a.wrapping_add(n).wrapping_add(1);\n"
        "let c = n.wrapping_mul(2654435761);\nlet d = n.wrapping_sub(16);\n"
        "outp[i] = b as f32 + (c >> 28) as f32 * 16.0 + (d >> 28) as f32 * 256.0;"
    ),
    "numeric-literals": Template(
        "let h: i32 = 0x1F + 0x0a;\nlet m = 0xFFu32 & (iin[i] * 37) as u32;\nlet e: f32 = 2.5e-1 + 1E2 + 0.5;\n"
        "outp[i] = h as f32 + m as f32 * 64.0 + e * 4.0;"
    ),
    "struct-array-member": Template(
        "let mut b = Bins { v: [0.0; 3], n: 0 };\nfor k in 0..3 {\n    b.v[k] = (k + 1) as f32;\n}\n"
        "b.n = iin[i];\nb.v[(iin[i] % 3) as usize] = fin[i];\n"
        "outp[i] = b.v[0] * 64.0 + b.v[1] * 8.0 + b.v[2] + b.n as f32 * 512.0;",
        "#[derive(Clone, Copy)]\nstruct Bins {\n    v: [f32; 3],\n    n: i32,\n}",
    ),
    "multiple-declarators": Template(
        "let (a, b): (f32, f32) = (fin[i], 2.0);\nlet c: f32;\nc = a * 4.0 + b;\nlet (j, k) = (1, iin[i]);\n"
        "outp[i] = c + (j + k) as f32 * 64.0;"
    ),
}

MOJO = {
    "vector-conditional": Template(
        "var a = $vec2(fin[i], 1.0)\nvar b = $vec2(2.0, fin[i] * 4.0)\nvar r = a if iin[i] > 7 else b\n"
        "outp[i] = r[0] * 16.0 + r[1]"
    ),
    "loop-return": Template(
        "outp[i] = find(iin[i], fin[i])",
        "def find(n: Int32, x: Float32) -> Float32:\n    for k in range(8):\n        if k == Int(n):\n"
        "            return x + Float32(k)\n    return -1.0",
    ),
    "unsigned-wraparound": Template(
        "var a: UInt32 = 4294967295\nvar n = UInt32(iin[i])\nvar b = a + n + 1\nvar c = n * 2654435761\n"
        "var d = n - 16\noutp[i] = Float32(b) + Float32(c >> 28) * 16.0 + Float32(d >> 28) * 256.0"
    ),
    "numeric-literals": Template(
        "var h: Int32 = 0x1F + 0X0a\nvar m = UInt32(0xFF) & UInt32(iin[i] * 37)\nvar e: Float32 = 2.5e-1 + 1E2 + .5\n"
        "outp[i] = Float32(h) + Float32(m) * 64.0 + e * 4.0"
    ),
    "struct-array-member": Template(
        "var b = Bins(Array[Float32, 3](fill=0.0), 0)\nfor k in range(3):\n    b.v[k] = Float32(k + 1)\n"
        "b.n = iin[i]\nb.v[Int(iin[i] % 3)] = fin[i]\n"
        "outp[i] = b.v[0] * 64.0 + b.v[1] * 8.0 + b.v[2] + Float32(b.n) * 512.0",
        "@fieldwise_init\nstruct Bins(Copyable, Movable):\n    var v: Array[Float32, 3]\n    var n: Int32",
    ),
    "multiple-declarators": Template(
        "var a, b = fin[i], Float32(2.0)\nvar c: Float32\nc = a * 4.0 + b\nvar j, k = Int32(1), iin[i]\n"
        "outp[i] = c + Float32(j + k) * 64.0"
    ),
}
