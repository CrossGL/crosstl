"""Conditional-side-effect, switch-return, loop-in-switch, implicit-conversion, matrix-compound-assignment and
diagonal-matrix probes in the languages that do not write them as the C family does.

HLSL multiplies matrices with ``mul``, where ``*`` and ``*=`` are element-wise, names elements by row and column, and
has no matrix constructor that takes one scalar. Metal passes references as ``thread`` references and has no ``*=``
on two matrices. GLSL ES 3.00 has no implicit conversions (GLSL ES 3.00.6, section 4.1.10). CUDA and HIP pass
references, mark device functions ``__device__`` and write ``unsigned int`` for ``uint``; OpenCL C passes pointers
and converts with casts. WGSL, Rust and Mojo convert explicitly; WGSL picks between calls with ``if``, because
``select`` evaluates both operands, and glam has no row-vector product, so the Rust probe multiplies by the
transpose. Mojo has no ``switch``, and CUDA, HIP, OpenCL C and Mojo have no matrix types.
"""

from __future__ import annotations

import re

from .base import ABSENT, C_FAMILY, PICK, STEPS, Template

_LITERAL = re.compile(r"\b(\d+\.\d+)\b")
_MATRICES = ("matrix-compound-assignment", "diagonal-matrix")


def _f(source: str) -> str:
    """``source`` with single-precision literals."""
    return _LITERAL.sub(r"\1f", source)


def _c(source: str) -> str:
    """``source`` with casts in place of conversions and single-precision literals."""
    return _f(source.replace("float(", "(float)("))


def _device(source: str) -> str:
    """``source`` with each function that returns float marked as a CUDA device function."""
    return "\n".join(
        f"__device__ {line}" if line.startswith("float ") else line
        for line in source.split("\n")
    )


HLSL = {
    "matrix-compound-assignment": Template(
        "float f = float(iin[i]);\nfloat2x2 m = float2x2(f, 2.0, 1.0, 3.0);\nfloat2x2 n = float2x2(1.0, 0.0, 2.0, 1.0);\n"
        "m = mul(m, n);\nm *= 2.0;\nm -= n;\nfloat2 v = float2(1.0, 1.0);\nv = mul(v, m);\n"
        "outp[i] = m[0][0] + m[1][0] * 16.0 + m[0][1] * 256.0 + m[1][1] * 4096.0 + v.x * 65536.0"
        " + v.y * 1048576.0;"
    ),
    "diagonal-matrix": Template(
        "float f = float(iin[i]);\nfloat2x2 d = float2x2(f, 0.0, 0.0, f);\n"
        "float2x2 e = float2x2(2.0, 0.0, 0.0, 2.0) + d;\n"
        "outp[i] = d[0][0] + d[1][0] * 16.0 + d[0][1] * 256.0 + d[1][1] * 4096.0 + e[1][1] * 65536.0"
        " + e[1][0] * 1048576.0;"
    ),
}

METAL = {
    "conditional-side-effect": Template(
        C_FAMILY["conditional-side-effect"].body,
        STEPS.replace("inout int k", "thread int& k"),
    ),
    "matrix-compound-assignment": Template(
        C_FAMILY["matrix-compound-assignment"].body.replace("m *= n;", "m = m * n;")
    ),
}

WEBGL = {
    "implicit-conversion": Template(
        "int a = iin[i];\nuint u = 3u;\nfloat x = float(a);\nfloat y = float(a) * 0.5 + 1.0;\n"
        "float z = x / 4.0 + float(u);\noutp[i] = x + y * 16.0 + z * 256.0;"
    ),
}

CUDA = {
    "conditional-side-effect": Template(
        _f(C_FAMILY["conditional-side-effect"].body),
        _device(_f(STEPS.replace("inout int k", "int& k"))),
    ),
    "switch-return": Template(C_FAMILY["switch-return"].body, _device(_f(PICK))),
    "implicit-conversion": Template(
        _f(C_FAMILY["implicit-conversion"].body).replace("uint", "unsigned int")
    ),
    **dict.fromkeys(_MATRICES, ABSENT),
}

OPENCL = {
    "conditional-side-effect": Template(
        "int k = iin[i];\nfloat r = (k & 1) == 0 ? bump(&k) : leap(&k);\nfloat s = k > 20 ? leap(&k) : 0.5f;\n"
        "outp[i] = r + (float)k * 64.0f + s * 4096.0f;",
        "float bump(int* k) {\n    *k = *k + 1;\n    return (float)*k;\n}\n\n"
        "float leap(int* k) {\n    *k = *k + 16;\n    return (float)*k * 2.0f;\n}",
    ),
    "switch-return": Template(_c(C_FAMILY["switch-return"].body), _c(PICK)),
    "loop-in-switch": Template(_c(C_FAMILY["loop-in-switch"].body)),
    "implicit-conversion": Template(_c(C_FAMILY["implicit-conversion"].body)),
    **dict.fromkeys(_MATRICES, ABSENT),
}

WGSL = {
    "conditional-side-effect": Template(
        "var k = iin[i];\nvar r: f32;\nif ((k & 1) == 0) {\n    r = bump(&k);\n} else {\n    r = leap(&k);\n}\n"
        "var s: f32 = 0.5;\nif (k > 20) {\n    s = leap(&k);\n}\noutp[i] = r + f32(k) * 64.0 + s * 4096.0;",
        "fn bump(k: ptr<function, i32>) -> f32 {\n    *k = *k + 1;\n    return f32(*k);\n}\n\n"
        "fn leap(k: ptr<function, i32>) -> f32 {\n    *k = *k + 16;\n    return f32(*k) * 2.0;\n}",
    ),
    "switch-return": Template(
        C_FAMILY["switch-return"].body,
        "fn pick(a: i32) -> f32 {\n    switch (a & 3) {\n        case 0: {\n            return 1.0;\n        }\n"
        "        case 1: {\n            return f32(a) * 2.0;\n        }\n        case 2: {\n        }\n"
        "        default: {\n            if (a > 8) {\n                return 64.0;\n            }\n        }\n    }\n"
        "    return 4.0 + f32(a);\n}",
    ),
    "loop-in-switch": Template(
        "var acc: f32 = 0.0;\nlet n = iin[i];\nswitch (n & 3) {\n    case 0: {\n"
        "        for (var k = 0; k < 8; k++) {\n            if (k == (n >> 2u)) {\n                break;\n"
        "            }\n            acc += 2.0;\n        }\n        acc += 1.0;\n    }\n    case 1: {\n"
        "        while (acc < 20.0) {\n            acc += 3.0;\n            if (acc > f32(n)) {\n                break;\n"
        "            }\n        }\n        acc += 128.0;\n    }\n    default: {\n"
        "        for (var k = 0; k < n; k++) {\n            if ((k & 1) == 1) {\n                continue;\n"
        "            }\n            acc += 4.0;\n        }\n        acc += 512.0;\n    }\n}\noutp[i] = acc + fin[i];"
    ),
    "implicit-conversion": Template(
        "let a = iin[i];\nlet u = 3u;\nlet x = f32(a);\nlet y = f32(a) * 0.5 + 1;\nlet z = x / 4 + f32(u);\n"
        "outp[i] = x + y * 16.0 + z * 256.0;"
    ),
    "matrix-compound-assignment": Template(
        "let f = f32(iin[i]);\nvar m = mat2x2<f32>(f, 1.0, 2.0, 3.0);\nlet n = mat2x2<f32>(1.0, 2.0, 0.0, 1.0);\n"
        "m *= n;\nm *= 2.0;\nm -= n;\nvar v = vec2<f32>(1.0, 1.0);\nv *= m;\n"
        "outp[i] = m[0][0] + m[0][1] * 16.0 + m[1][0] * 256.0 + m[1][1] * 4096.0 + v.x * 65536.0"
        " + v.y * 1048576.0;"
    ),
    "diagonal-matrix": Template(
        "let f = f32(iin[i]);\nlet d = mat2x2<f32>(f, 0.0, 0.0, f);\n"
        "let e = mat2x2<f32>(2.0, 0.0, 0.0, 2.0) + d;\n"
        "outp[i] = d[0][0] + d[0][1] * 16.0 + d[1][0] * 256.0 + d[1][1] * 4096.0 + e[1][1] * 65536.0"
        " + e[0][1] * 1048576.0;"
    ),
}

RUST = {
    "conditional-side-effect": Template(
        "let mut k = iin[i];\nlet r = if (k & 1) == 0 { bump(&mut k) } else { leap(&mut k) };\n"
        "let s = if k > 20 { leap(&mut k) } else { 0.5 };\noutp[i] = r + k as f32 * 64.0 + s * 4096.0;",
        "fn bump(k: &mut i32) -> f32 {\n    *k += 1;\n    *k as f32\n}\n\n"
        "fn leap(k: &mut i32) -> f32 {\n    *k += 16;\n    *k as f32 * 2.0\n}",
    ),
    "switch-return": Template(
        C_FAMILY["switch-return"].body,
        "fn pick(a: i32) -> f32 {\n    match a & 3 {\n        0 => return 1.0,\n        1 => return a as f32 * 2.0,\n"
        "        2 => {}\n        _ => {\n            if a > 8 {\n                return 64.0;\n            }\n        }\n"
        "    }\n    4.0 + a as f32\n}",
    ),
    "loop-in-switch": Template(
        "let mut acc = 0.0;\nlet n = iin[i];\nmatch n & 3 {\n    0 => {\n        for k in 0..8 {\n"
        "            if k == (n >> 2) {\n                break;\n            }\n            acc += 2.0;\n        }\n"
        "        acc += 1.0;\n    }\n    1 => {\n        while acc < 20.0 {\n            acc += 3.0;\n"
        "            if acc > n as f32 {\n                break;\n            }\n        }\n        acc += 128.0;\n"
        "    }\n    _ => {\n        for k in 0..n {\n            if (k & 1) == 1 {\n                continue;\n"
        "            }\n            acc += 4.0;\n        }\n        acc += 512.0;\n    }\n}\noutp[i] = acc + fin[i];"
    ),
    "implicit-conversion": Template(
        "let a = iin[i];\nlet u: u32 = 3;\nlet x = a as f32;\nlet y = a as f32 * 0.5 + 1.0;\n"
        "let z = x / 4.0 + u as f32;\noutp[i] = x + y * 16.0 + z * 256.0;"
    ),
    "matrix-compound-assignment": Template(
        "let f = iin[i] as f32;\nlet mut m = $mat2::from_cols($vec2::new(f, 1.0), $vec2::new(2.0, 3.0));\n"
        "let n = $mat2::from_cols($vec2::new(1.0, 2.0), $vec2::new(0.0, 1.0));\n"
        "m *= n;\nm *= 2.0;\nm -= n;\nlet v = m.transpose() * $vec2::new(1.0, 1.0);\n"
        "outp[i] = m.x_axis.x + m.x_axis.y * 16.0 + m.y_axis.x * 256.0 + m.y_axis.y * 4096.0 + v.x * 65536.0"
        " + v.y * 1048576.0;"
    ),
    "diagonal-matrix": Template(
        "let f = iin[i] as f32;\nlet d = $mat2::from_diagonal($vec2::splat(f));\n"
        "let e = $mat2::from_diagonal($vec2::splat(2.0)) + d;\n"
        "outp[i] = d.x_axis.x + d.x_axis.y * 16.0 + d.y_axis.x * 256.0 + d.y_axis.y * 4096.0"
        " + e.y_axis.y * 65536.0 + e.x_axis.y * 1048576.0;"
    ),
}

MOJO = {
    "conditional-side-effect": Template(
        "var k: Int32 = iin[i]\nvar r = bump(k) if (k & 1) == 0 else leap(k)\n"
        "var s: Float32 = leap(k) if k > 20 else 0.5\noutp[i] = r + Float32(k) * 64.0 + s * 4096.0",
        "def bump(mut k: Int32) -> Float32:\n    k = k + 1\n    return Float32(k)\n\n\n"
        "def leap(mut k: Int32) -> Float32:\n    k = k + 16\n    return Float32(k) * 2.0",
    ),
    "switch-return": ABSENT,
    "loop-in-switch": ABSENT,
    "implicit-conversion": Template(
        "var a = iin[i]\nvar u: UInt32 = 3\nvar x = Float32(a)\nvar y = Float32(a) * 0.5 + 1.0\n"
        "var z = x / 4.0 + Float32(u)\noutp[i] = x + y * 16.0 + z * 256.0"
    ),
    **dict.fromkeys(_MATRICES, ABSENT),
}
