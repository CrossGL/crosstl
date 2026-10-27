"""Switch-in-loop, do-while-continue, chained-conditional, precedence, matrix-arithmetic and vector-array probes in
the languages that do not write them as the C family does.

CUDA and HIP build vectors with ``make_float2``; OpenCL C converts with casts and builds vectors with vector literals.
WGSL picks a value with ``select``, writes a do-while loop as a ``loop`` whose ``continuing`` block tests the
condition, needs shift amounts of type ``u32`` and does not let shifts and bitwise operators share an expression
without parentheses (WGSL, section 8.19). Rust and Mojo choose with ``if`` expressions and have no do-while loop; Mojo
has no ``switch`` either. CUDA, HIP, OpenCL C and Mojo have no matrix types.
"""

from __future__ import annotations

import re

from .base import ABSENT, C_FAMILY, Template

_LITERAL = re.compile(r"\b(\d+\.\d+)\b")


def _c(name: str) -> str:
    """The C family's probe with casts in place of conversions and single-precision literals."""
    return _LITERAL.sub(r"\1f", C_FAMILY[name].body.replace("float(", "(float)("))


_CUDA_POINTS = (
    "float2 pts[3];\nfor (int k = 0; k < 3; k++) {\n    pts[k] = make_float2(fin[i] + float(k), float(iin[i] * k));\n}\n"
    "int j = iin[i] % 3;\npts[j].y = pts[j].y + 100.0f;\n"
    "outp[i] = pts[0].x + pts[1].y * 10.0f + pts[2].x * 1024.0f + pts[j].y;"
)

CUDA = {
    "matrix-arithmetic": ABSENT,
    "vector-array": Template(_CUDA_POINTS),
}

OPENCL = {
    "switch-in-loop": Template(_c("switch-in-loop")),
    "do-while-continue": Template(_c("do-while-continue")),
    "chained-conditional": Template(_c("chained-conditional")),
    "operator-precedence": Template(_c("operator-precedence")),
    "matrix-arithmetic": ABSENT,
    "vector-array": Template(
        _c("vector-array").replace("$vec2(", "(float2)(").replace("$vec2", "float2")
    ),
}

WGSL = {
    "switch-in-loop": Template(
        "var acc: f32 = 0.0;\nfor (var k = 0; k < 6; k++) {\n    switch ((iin[i] + k) & 3) {\n"
        "        case 0: {\n            acc += 1.0;\n        }\n        case 1: {\n            continue;\n        }\n"
        "        case 2: {\n            acc += fin[i];\n        }\n        default: {\n            acc *= 2.0;\n"
        "        }\n    }\n    acc += 0.5;\n}\noutp[i] = acc;"
    ),
    "do-while-continue": Template(
        "let m = 5 + (iin[i] & 3);\nvar k = 0;\nvar acc = fin[i];\nloop {\n    k++;\n    if (k > 20) {\n"
        "        break;\n    }\n    if (k > 4) {\n        acc += 0.5;\n        continue;\n    }\n"
        "    acc += f32(k);\n    continuing {\n        break if k >= m;\n    }\n}\noutp[i] = acc;"
    ),
    "chained-conditional": Template(
        "let a = iin[i];\nlet b = select(3.0, select(2.0, 1.0, a > 9), a > 1);\n"
        "let c = select(5.0 + fin[i], 4.0, a > 6);\n"
        "outp[i] = select(select(select(-fin[i], 3.0, a < 12), fin[i] * 2.0, a < 8), fin[i], a < 4)"
        " + b * 10.0 + c * 256.0;"
    ),
    "operator-precedence": Template(
        "let a = iin[i];\nlet r = ((((a + 3) << 2u) & 60) ^ (a * 2)) | 1;\nlet s = 20 - a - 3 + a / 2 * 3;\n"
        "outp[i] = f32(r) + f32(s) * 64.0 + fin[i] / 2.0 * 4.0 - 1.0 - 2.0;"
    ),
    "matrix-arithmetic": Template(
        "let m = mat2x2<f32>(fin[i], 1.0, 2.0, 3.0);\nlet n = mat2x2<f32>(4.0, f32(iin[i]), 5.0, 6.0);\n"
        "var r = (m + n) * 2.0 - 0.5 * n;\nr += m;\n"
        "outp[i] = r[0][0] + r[0][1] * 10.0 + r[1][0] * 100.0 + r[1][1] * 1000.0;"
    ),
    "vector-array": Template(
        "var pts: array<vec2<f32>, 3>;\nfor (var k = 0; k < 3; k++) {\n"
        "    pts[k] = vec2<f32>(fin[i] + f32(k), f32(iin[i] * k));\n}\n"
        "let j = iin[i] % 3;\npts[j].y = pts[j].y + 100.0;\n"
        "outp[i] = pts[0].x + pts[1].y * 10.0 + pts[2].x * 1024.0 + pts[j].y;"
    ),
}

RUST = {
    "switch-in-loop": Template(
        "let mut acc = 0.0;\nfor k in 0..6 {\n    match (iin[i] + k) & 3 {\n        0 => acc += 1.0,\n"
        "        1 => continue,\n        2 => acc += fin[i],\n        _ => acc *= 2.0,\n    }\n    acc += 0.5;\n}\n"
        "outp[i] = acc;"
    ),
    "do-while-continue": ABSENT,
    "chained-conditional": Template(
        "let a = iin[i];\nlet b = if a > 1 { if a > 9 { 1.0 } else { 2.0 } } else { 3.0 };\n"
        "let c = if a > 6 { 4.0 } else { 5.0 + fin[i] };\n"
        "outp[i] = (if a < 4 { fin[i] } else if a < 8 { fin[i] * 2.0 } else if a < 12 { 3.0 } else { -fin[i] })"
        " + b * 10.0 + c * 256.0;"
    ),
    "operator-precedence": Template(
        "let a = iin[i];\nlet r = a + 3 << 2 & 60 ^ a * 2 | 1;\nlet s = 20 - a - 3 + a / 2 * 3;\n"
        "outp[i] = r as f32 + s as f32 * 64.0 + fin[i] / 2.0 * 4.0 - 1.0 - 2.0;"
    ),
    "matrix-arithmetic": Template(
        "let m = $mat2::from_cols($vec2::new(fin[i], 1.0), $vec2::new(2.0, 3.0));\n"
        "let n = $mat2::from_cols($vec2::new(4.0, iin[i] as f32), $vec2::new(5.0, 6.0));\n"
        "let mut r = (m + n) * 2.0 - 0.5 * n;\nr += m;\n"
        "outp[i] = r.x_axis.x + r.x_axis.y * 10.0 + r.y_axis.x * 100.0 + r.y_axis.y * 1000.0;"
    ),
    "vector-array": Template(
        "let mut pts = [$vec2::ZERO; 3];\nfor k in 0..3 {\n"
        "    pts[k] = $vec2::new(fin[i] + k as f32, (iin[i] * k as i32) as f32);\n}\n"
        "let j = (iin[i] % 3) as usize;\npts[j].y = pts[j].y + 100.0;\n"
        "outp[i] = pts[0].x + pts[1].y * 10.0 + pts[2].x * 1024.0 + pts[j].y;"
    ),
}

MOJO = {
    "switch-in-loop": ABSENT,
    "do-while-continue": ABSENT,
    "chained-conditional": Template(
        "var a = iin[i]\nvar b: Float32 = (Float32(1.0) if a > 9 else 2.0) if a > 1 else 3.0\n"
        "var c: Float32 = Float32(4.0) if a > 6 else 5.0 + fin[i]\n"
        "outp[i] = (fin[i] if a < 4 else fin[i] * 2.0 if a < 8 else Float32(3.0) if a < 12 else -fin[i])"
        " + b * 10.0 + c * 256.0"
    ),
    "operator-precedence": Template(
        "var a = iin[i]\nvar r = a + 3 << 2 & 60 ^ a * 2 | 1\nvar s = 20 - a - 3 + a // 2 * 3\n"
        "outp[i] = Float32(r) + Float32(s) * 64.0 + fin[i] / 2.0 * 4.0 - 1.0 - 2.0"
    ),
    "matrix-arithmetic": ABSENT,
    "vector-array": Template(
        "var pts = Array[$vec2, 3](fill=$vec2(0.0, 0.0))\nfor k in range(3):\n"
        "    pts[k] = $vec2(fin[i] + Float32(k), Float32(Int(iin[i]) * k))\n"
        "var j = Int(iin[i] % 3)\npts[j][1] = pts[j][1] + 100.0\n"
        "outp[i] = pts[0][0] + pts[1][1] * 10.0 + pts[2][0] * 1024.0 + pts[j][1]"
    ),
}
