"""Composite-type probes in the languages that do not write them as the C family does (:data:`.base.C_FAMILY`).

HLSL indexes a matrix by rows, WGSL and Rust can only read swizzles, OpenCL C converts with casts, and CUDA, HIP and
Mojo have no swizzles or matrices.
"""

from __future__ import annotations

from .base import ABSENT, C_FAMILY, Template

_NESTED_BODY = C_FAMILY["nested-struct"].body

# HLSL fills a matrix from scalars by rows, so float2x2(f, 2.0, 1.0, 3.0) is CrossGL's mat2(f, 1.0, 2.0, 3.0), and
# CrossGL's column k is the k-th element of each HLSL row.
HLSL = {
    "matrix-indexing": Template(
        "float f = float(iin[i]);\nfloat2x2 m = float2x2(f, 2.0, 1.0, 3.0);\nint k = iin[i] & 1;\n"
        "float2 c = float2(m[0][k], m[1][k]);\noutp[i] = m[0][1] * 100.0 + m[1].x * 10.0 + c.x + 0.5 * c.y;"
    ),
}

WEBGL = {
    # GLSL ES 3.00 has no initializer lists.
    "array-initializer": Template(
        "float values[4] = float[4](1.0, 2.0, 4.0, 8.0);\n"
        "outp[i] = values[iin[i] & 3] + float(iin[i]);"
    ),
}

CUDA = {"swizzle-assignment": ABSENT, "matrix-indexing": ABSENT}

OPENCL = {
    "array-initializer": Template(
        "float values[4] = {1.0f, 2.0f, 4.0f, 8.0f};\noutp[i] = values[iin[i] & 3] + (float)iin[i];"
    ),
    "nested-struct": Template(
        _NESTED_BODY.replace("float(iin[i])", "(float)iin[i]"),
        "typedef struct {\n    float a;\n    float b;\n} Pair;\n\n"
        "typedef struct {\n    Pair p;\n    float w;\n} Frame;",
    ),
    "swizzle-assignment": Template(
        "float4 v = (float4)(0.0f, 0.0f, 0.0f, 0.0f);\nv.y = (float)iin[i];\nv.zx = (float2)(2.0f, 3.0f);\n"
        "v.wy += (float2)(1.0f, 0.5f);\noutp[i] = v.x * 1000.0f + v.y * 100.0f + v.z * 10.0f + v.w;"
    ),
    "matrix-indexing": ABSENT,
}

WGSL = {
    "array-initializer": Template(
        "var values = array<f32, 4>(1.0, 2.0, 4.0, 8.0);\noutp[i] = values[iin[i] & 3] + f32(iin[i]);"
    ),
    "nested-struct": Template(
        "var s: Frame;\ns.p.a = f32(iin[i]);\ns.p.b = 2.0;\ns.w = 3.0;\noutp[i] = s.p.a * s.w + s.p.b;",
        "struct Pair {\n    a: f32,\n    b: f32,\n}\n\nstruct Frame {\n    p: Pair,\n    w: f32,\n}",
    ),
    # A WGSL swizzle is a value, never a reference, so only single components can be assigned.
    "swizzle-assignment": ABSENT,
    "matrix-indexing": Template(
        "let f = f32(iin[i]);\nvar m = mat2x2<f32>(f, 1.0, 2.0, 3.0);\nlet c = m[iin[i] & 1];\n"
        "outp[i] = m[1][0] * 100.0 + m[0].y * 10.0 + c.x + 0.5 * c.y;"
    ),
}

RUST = {
    "array-initializer": Template(
        "let values: [f32; 4] = [1.0, 2.0, 4.0, 8.0];\n"
        "outp[i] = values[(iin[i] & 3) as usize] + iin[i] as f32;"
    ),
    "nested-struct": Template(
        "let mut s = Frame::default();\ns.p.a = iin[i] as f32;\ns.p.b = 2.0;\ns.w = 3.0;\n"
        "outp[i] = s.p.a * s.w + s.p.b;",
        "#[derive(Clone, Copy, Default)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}\n\n"
        "#[derive(Clone, Copy, Default)]\nstruct Frame {\n    p: Pair,\n    w: f32,\n}",
    ),
    # glam's swizzles are read-only methods.
    "swizzle-assignment": ABSENT,
    "matrix-indexing": Template(
        "let f = iin[i] as f32;\nlet m = $mat2::from_cols($vec2::new(f, 1.0), $vec2::new(2.0, 3.0));\n"
        "let c = m.col((iin[i] & 1) as usize);\n"
        "outp[i] = m.col(1).x * 100.0 + m.col(0).y * 10.0 + c.x + 0.5 * c.y;"
    ),
}

MOJO = {
    "array-initializer": Template(
        "var values: Array[Float32, 4] = [1.0, 2.0, 4.0, 8.0]\n"
        "outp[i] = values[Int(iin[i] & 3)] + Float32(iin[i])"
    ),
    "nested-struct": Template(
        "var s = Frame(Pair(0.0, 0.0), 0.0)\ns.p.a = Float32(iin[i])\ns.p.b = 2.0\ns.w = 3.0\n"
        "outp[i] = s.p.a * s.w + s.p.b",
        "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32\n\n\n"
        "@fieldwise_init\nstruct Frame(Copyable, Movable):\n    var p: Pair\n    var w: Float32",
    ),
    "swizzle-assignment": ABSENT,
    "matrix-indexing": ABSENT,
}
