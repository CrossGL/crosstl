"""Nested-array, dynamic-index, block-scope, private-global and integer-vector probes in the languages that do not
write them as the C family does.

GLSL ES 3.00 declares one-dimensional arrays only. HLSL marks a private global ``static``, since an unqualified one
is a uniform, and WGSL declares it ``var<private>``. Metal, CUDA, HIP, OpenCL C, Rust and Mojo have no global that
each invocation holds a copy of: their mutable globals are one variable shared by every thread, and Metal allows none
at program scope outside the constant address space. CUDA and HIP vector types are plain structs with neither
subscripts nor operators. WGSL shifts by unsigned amounts, Rust indexes with ``usize``, and Mojo, which has no bare
blocks, opens its inner scope with ``if True:``.
"""

from __future__ import annotations

from .base import ABSENT, C_FAMILY, Template

_NO_PRIVATE_GLOBAL = {"global-variable": ABSENT}

WEBGL = {"array-of-arrays": ABSENT}

HLSL = {
    "global-variable": Template(
        C_FAMILY["global-variable"].body,
        "static " + C_FAMILY["global-variable"].declarations,
    ),
}

METAL = dict(_NO_PRIVATE_GLOBAL)

CUDA = {
    **_NO_PRIVATE_GLOBAL,
    "vector-dynamic-index": ABSENT,
    "integer-vector-arithmetic": ABSENT,
}

OPENCL = {
    **_NO_PRIVATE_GLOBAL,
    "array-of-arrays": Template(
        "float a[2][3];\nfor (int r = 0; r < 2; r++) {\n    for (int c = 0; c < 3; c++) {\n"
        "        a[r][c] = (float)(r * 3 + c);\n    }\n}\na[iin[i] & 1][2] = fin[i];\n"
        "outp[i] = a[1][2] * 100.0f + a[0][2] * 10.0f + a[1][0];"
    ),
    "vector-dynamic-index": Template(
        "float4 v = (float4)(1.0f, 2.0f, 4.0f, 8.0f);\nint k = iin[i] & 3;\nv[k] = fin[i];\n"
        "outp[i] = v[k] * 100.0f + v[(k + 1) & 3] * 10.0f + v.x;"
    ),
    "integer-vector-arithmetic": Template(
        "int2 a = (int2)(iin[i], 3);\nint2 b = a * 2 + (int2)(1, 5);\n"
        "int2 c = (b >> 1) ^ (b & (int2)(6, 6));\nint2 d = b / 3 + b % 4;\n"
        "outp[i] = (float)(c.x * 100 + c.y * 10 + d.x + d.y * 1000);"
    ),
    "block-scope": Template(
        "float x = fin[i];\nfloat r = 0.0f;\n{\n    float x = 2.0f;\n    r += x;\n}\n"
        "for (int k = 0; k < 2; k++) {\n    float x = (float)(k + 1) * 10.0f;\n    r += x;\n}\n"
        "outp[i] = r * 100.0f + x;"
    ),
}

WGSL = {
    "array-of-arrays": Template(
        "var a: array<array<f32, 3>, 2>;\nfor (var r = 0; r < 2; r++) {\n    for (var c = 0; c < 3; c++) {\n"
        "        a[r][c] = f32(r * 3 + c);\n    }\n}\na[iin[i] & 1][2] = fin[i];\n"
        "outp[i] = a[1][2] * 100.0 + a[0][2] * 10.0 + a[1][0];"
    ),
    "vector-dynamic-index": Template(
        "var v = vec4<f32>(1.0, 2.0, 4.0, 8.0);\nlet k = iin[i] & 3;\nv[k] = fin[i];\n"
        "outp[i] = v[k] * 100.0 + v[(k + 1) & 3] * 10.0 + v.x;"
    ),
    "integer-vector-arithmetic": Template(
        "let a = vec2<i32>(iin[i], 3);\nlet b = a * 2 + vec2<i32>(1, 5);\n"
        "let c = (b >> vec2<u32>(1u)) ^ (b & vec2<i32>(6, 6));\nlet d = b / 3 + b % 4;\n"
        "outp[i] = f32(c.x * 100 + c.y * 10 + d.x + d.y * 1000);"
    ),
    "block-scope": Template(
        "let x = fin[i];\nvar r = 0.0;\n{\n    let x = 2.0;\n    r += x;\n}\n"
        "for (var k = 0; k < 2; k++) {\n    let x = f32(k + 1) * 10.0;\n    r += x;\n}\n"
        "outp[i] = r * 100.0 + x;"
    ),
    "global-variable": Template(
        "let before = counter;\ncounter = fin[i];\nbump();\noutp[i] = counter * 10.0 + before;",
        "var<private> counter: f32 = 1.0;\n\nfn bump() {\n    counter = counter * 2.0 + 1.0;\n}",
    ),
}

RUST = {
    **_NO_PRIVATE_GLOBAL,
    "array-of-arrays": Template(
        "let mut a = [[0.0f32; 3]; 2];\nfor r in 0..2 {\n    for c in 0..3 {\n"
        "        a[r][c] = (r * 3 + c) as f32;\n    }\n}\na[(iin[i] & 1) as usize][2] = fin[i];\n"
        "outp[i] = a[1][2] * 100.0 + a[0][2] * 10.0 + a[1][0];"
    ),
    "vector-dynamic-index": Template(
        "let mut v = $vec4::new(1.0, 2.0, 4.0, 8.0);\nlet k = (iin[i] & 3) as usize;\nv[k] = fin[i];\n"
        "outp[i] = v[k] * 100.0 + v[(k + 1) & 3] * 10.0 + v.x;"
    ),
    "integer-vector-arithmetic": Template(
        "let a = $ivec2::new(iin[i], 3);\nlet b = a * 2 + $ivec2::new(1, 5);\n"
        "let c = (b >> 1) ^ (b & $ivec2::splat(6));\nlet d = b / 3 + b % 4;\n"
        "outp[i] = (c.x * 100 + c.y * 10 + d.x + d.y * 1000) as f32;"
    ),
    "block-scope": Template(
        "let x = fin[i];\nlet mut r = 0.0;\n{\n    let x = 2.0;\n    r += x;\n}\n"
        "for k in 0..2 {\n    let x = (k + 1) as f32 * 10.0;\n    r += x;\n}\n"
        "outp[i] = r * 100.0 + x;"
    ),
}

MOJO = {
    **_NO_PRIVATE_GLOBAL,
    "array-of-arrays": Template(
        "var a = Array[Array[Float32, 3], 2](fill=Array[Float32, 3](fill=0.0))\nfor r in range(2):\n"
        "    for c in range(3):\n        a[r][c] = Float32(r * 3 + c)\na[Int(iin[i] & 1)][2] = fin[i]\n"
        "outp[i] = a[1][2] * 100.0 + a[0][2] * 10.0 + a[1][0]"
    ),
    "vector-dynamic-index": Template(
        "var v = $vec4(1.0, 2.0, 4.0, 8.0)\nvar k = Int(iin[i] & 3)\nv[k] = fin[i]\n"
        "outp[i] = v[k] * 100.0 + v[(k + 1) & 3] * 10.0 + v[0]"
    ),
    "integer-vector-arithmetic": Template(
        "var a = $ivec2(iin[i], 3)\nvar b = a * 2 + $ivec2(1, 5)\n"
        "var c = (b >> 1) ^ (b & $ivec2(6, 6))\nvar d = b // 3 + b % 4\n"
        "outp[i] = Float32(c[0] * 100 + c[1] * 10 + d[0] + d[1] * 1000)"
    ),
    "block-scope": Template(
        "var x = fin[i]\nvar r: Float32 = 0.0\nif True:\n    var x: Float32 = 2.0\n    r += x\n"
        "for k in range(2):\n    var x = Float32(k + 1) * 10.0\n    r += x\n"
        "outp[i] = r * 100.0 + x"
    ),
}
