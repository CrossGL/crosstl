"""Nested-switch, unsigned-switch, multi-variable-loop, forward-declaration, non-square-matrix and
struct-vector-member probes in the languages that do not write them as the C family does.

HLSL names a matrix by rows and columns, so CrossGL's two-column, three-row ``mat2x3`` is ``float3x2``, filled row by
row and multiplied with ``mul``. CUDA and HIP build vectors with ``make_float4``, have no swizzles, mark device
functions ``__device__`` and write ``unsigned int`` for ``uint``; OpenCL C converts with casts and builds vectors with
vector literals. WGSL has no fallthrough, comma operator or prototypes, needs a ``default`` clause in every switch and
writes a multi-component swizzle one component at a time; a function may be called above its definition, as in Rust
and Mojo. Rust and Mojo step two counters in a ``while`` loop. Mojo has no ``switch``, and CUDA, HIP, OpenCL C, Mojo
and Rust's glam have no non-square matrices.
"""

from __future__ import annotations

import re

from .base import ABSENT, C_FAMILY, QUAD, TWICE, Template

_LITERAL = re.compile(r"\b(\d+\.\d+)\b")


def _c(name: str) -> str:
    """The C family's probe with casts in place of conversions and single-precision literals."""
    body = C_FAMILY[name].body.replace("float(", "(float)(").replace("uint(", "(uint)(")
    return _LITERAL.sub(r"\1f", body)


_POSE_SUM = "p.v.x + p.v.y * 16.0 + p.v.z * 128.0 + p.v.w * 4096.0 + p.u.x * 16384.0 + p.u.y * 65536.0"
_C_POSE_SUM = _LITERAL.sub(r"\1f", _POSE_SUM)
_C_POSE_BODY = f"p.n = iin[i] & 3;\noutp[i] = {_C_POSE_SUM} + (float)p.n * 1048576.0f;"
_C_QUAD = _LITERAL.sub(r"\1f", QUAD + TWICE)

HLSL = {
    "non-square-matrix": Template(
        "float a = float(iin[i]);\nfloat3x2 m = float3x2(a, 3.0, 1.0, 4.0, 2.0, 5.0);\n"
        "float3 v = mul(m, float2(1.0, 2.0));\nfloat2 w = mul(float3(1.0, 1.0, 2.0), m);\n"
        "outp[i] = v.x + v.z * 16.0 + w.x * 256.0 + w.y * 4096.0 + m[2][1] * 65536.0;"
    ),
}

CUDA = {
    "unsigned-switch": Template(_c("unsigned-switch").replace("uint", "unsigned int")),
    "forward-declaration": Template(
        C_FAMILY["forward-declaration"].body,
        "\n".join(
            f"__device__ {line}" if line.startswith("float ") else line
            for line in _C_QUAD.split("\n")
        ),
    ),
    "non-square-matrix": ABSENT,
    "struct-vector-member": Template(
        "Pose p;\np.v = make_float4((float)iin[i], 1.0f, 2.0f, 3.0f);\np.u = make_float2(p.v.w, p.v.x);\n"
        "p.v.y = p.u.x * 2.0f;\np.v.z = p.u.y * 2.0f;\np.u.y = p.v.y + 1.0f;\n"
        + _C_POSE_BODY,
        "struct Pose {\n    float4 v;\n    float2 u;\n    int n;\n};",
    ),
}

OPENCL = {
    "nested-switch": Template(_c("nested-switch")),
    "unsigned-switch": Template(_c("unsigned-switch")),
    "multi-variable-loop": Template(_c("multi-variable-loop")),
    "forward-declaration": Template(_c("forward-declaration"), _C_QUAD),
    "non-square-matrix": ABSENT,
    "struct-vector-member": Template(
        "Pose p;\np.v = (float4)((float)iin[i], 1.0f, 2.0f, 3.0f);\np.u = p.v.wx;\n"
        "p.v.yz = p.u * 2.0f;\np.u.y = p.v.y + 1.0f;\n" + _C_POSE_BODY,
        "typedef struct {\n    float4 v;\n    float2 u;\n    int n;\n} Pose;",
    ),
}

WGSL = {
    "nested-switch": Template(
        "var acc: f32 = 0.0;\nlet a = iin[i];\nswitch (a & 3) {\n    case 0: {\n        switch ((a >> 2u) & 1) {\n"
        "            case 0: {\n                acc += 1.0;\n            }\n            default: {\n"
        "                acc += 2.0;\n            }\n        }\n        acc += 4.0;\n    }\n    case 1: {\n"
        "        switch (a >> 2u) {\n            case 1: {\n                acc += 8.0;\n                acc += 16.0;\n"
        "            }\n            case 2: {\n                acc += 16.0;\n            }\n            default: {\n"
        "            }\n        }\n        acc += 32.0;\n    }\n    default: {\n        acc += 64.0;\n    }\n}\n"
        "outp[i] = acc + fin[i];"
    ),
    "unsigned-switch": Template(
        "let u = u32(iin[i] - 8);\nvar r: f32 = 0.0;\nswitch (u) {\n    case 0u: {\n        r = 1.0;\n    }\n"
        "    case 4294967295u: {\n        r = 2.0;\n    }\n    case 4294967288u: {\n        r = 3.0;\n    }\n"
        "    case 7u: {\n        r = 4.0;\n    }\n    default: {\n        r = f32(u % 5u) * 8.0;\n    }\n}\n"
        "outp[i] = r + fin[i];"
    ),
    "multi-variable-loop": Template(
        "var acc: f32 = 0.0;\nlet n = 6 + (iin[i] & 3);\nvar a = 0;\nvar b = n;\nloop {\n    if (a >= b) {\n"
        "        break;\n    }\n    if (a == 1) {\n        continue;\n    }\n    acc += f32(a * 16 + b);\n"
        "    continuing {\n        a++;\n        b--;\n    }\n}\noutp[i] = acc + fin[i];"
    ),
    "forward-declaration": Template(
        "outp[i] = quadruple(fin[i]) + quadruple(f32(iin[i]));",
        "fn quadruple(x: f32) -> f32 {\n    return twice(twice(x));\n}\n\nfn twice(x: f32) -> f32 {\n    return x * 2.0;\n}",
    ),
    "non-square-matrix": Template(
        "let a = f32(iin[i]);\nlet m = mat2x3<f32>(a, 1.0, 2.0, 3.0, 4.0, 5.0);\n"
        "let v = m * vec2<f32>(1.0, 2.0);\nlet w = vec3<f32>(1.0, 1.0, 2.0) * m;\n"
        "outp[i] = v.x + v.z * 16.0 + w.x * 256.0 + w.y * 4096.0 + m[1].z * 65536.0;"
    ),
    "struct-vector-member": Template(
        "var p: Pose;\np.v = vec4<f32>(f32(iin[i]), 1.0, 2.0, 3.0);\np.u = p.v.wx;\np.n = iin[i] & 3;\n"
        "let s = p.u * 2.0;\np.v.y = s.x;\np.v.z = s.y;\np.u.y = p.v.y + 1.0;\n"
        f"outp[i] = {_POSE_SUM} + f32(p.n) * 1048576.0;",
        "struct Pose {\n    v: vec4<f32>,\n    u: vec2<f32>,\n    n: i32,\n}",
    ),
}

RUST = {
    "nested-switch": Template(
        "let mut acc = 0.0;\nlet a = iin[i];\nmatch a & 3 {\n    0 => {\n        match (a >> 2) & 1 {\n"
        "            0 => acc += 1.0,\n            _ => acc += 2.0,\n        }\n        acc += 4.0;\n    }\n"
        "    1 => {\n        match a >> 2 {\n            1 => {\n                acc += 8.0;\n                acc += 16.0;\n"
        "            }\n            2 => acc += 16.0,\n            _ => {}\n        }\n        acc += 32.0;\n    }\n"
        "    _ => acc += 64.0,\n}\noutp[i] = acc + fin[i];"
    ),
    "unsigned-switch": Template(
        "let u = (iin[i] - 8) as u32;\nlet r = match u {\n    0 => 1.0,\n    4294967295 => 2.0,\n"
        "    4294967288 => 3.0,\n    7 => 4.0,\n    _ => (u % 5) as f32 * 8.0,\n};\noutp[i] = r + fin[i];"
    ),
    "multi-variable-loop": Template(
        "let mut acc = 0.0;\nlet n = 6 + (iin[i] & 3);\nlet (mut a, mut b) = (0, n);\nwhile a < b {\n"
        "    if a != 1 {\n        acc += (a * 16 + b) as f32;\n    }\n    a += 1;\n    b -= 1;\n}\n"
        "outp[i] = acc + fin[i];"
    ),
    "forward-declaration": Template(
        "outp[i] = quadruple(fin[i]) + quadruple(iin[i] as f32);",
        "fn quadruple(x: f32) -> f32 {\n    twice(twice(x))\n}\n\nfn twice(x: f32) -> f32 {\n    x * 2.0\n}",
    ),
    "non-square-matrix": ABSENT,
    "struct-vector-member": Template(
        "let mut p = Pose {\n    v: $vec4::new(iin[i] as f32, 1.0, 2.0, 3.0),\n    u: $vec2::ZERO,\n    n: 0,\n};\n"
        "p.u = $vec2::new(p.v.w, p.v.x);\np.n = iin[i] & 3;\nlet s = p.u * 2.0;\np.v.y = s.x;\np.v.z = s.y;\n"
        f"p.u.y = p.v.y + 1.0;\noutp[i] = {_POSE_SUM} + p.n as f32 * 1048576.0;",
        "#[derive(Clone, Copy)]\nstruct Pose {\n    v: $vec4,\n    u: $vec2,\n    n: i32,\n}",
    ),
}

MOJO = {
    "nested-switch": ABSENT,
    "unsigned-switch": ABSENT,
    "multi-variable-loop": Template(
        "var acc: Float32 = 0.0\nvar n = 6 + (iin[i] & 3)\nvar a: Int32 = 0\nvar b = n\nwhile a < b:\n"
        "    if a != 1:\n        acc += Float32(a * 16 + b)\n    a += 1\n    b -= 1\noutp[i] = acc + fin[i]"
    ),
    "forward-declaration": Template(
        "outp[i] = quadruple(fin[i]) + quadruple(Float32(iin[i]))",
        "def quadruple(x: Float32) -> Float32:\n    return twice(twice(x))\n\n\n"
        "def twice(x: Float32) -> Float32:\n    return x * 2.0",
    ),
    "non-square-matrix": ABSENT,
    "struct-vector-member": Template(
        "var p = Pose($vec4(Float32(iin[i]), 1.0, 2.0, 3.0), $vec2(0.0, 0.0), 0)\n"
        "p.u = $vec2(p.v[3], p.v[0])\np.n = iin[i] & 3\nvar s = p.u * 2.0\np.v[1] = s[0]\np.v[2] = s[1]\n"
        "p.u[1] = p.v[1] + 1.0\noutp[i] = p.v[0] + p.v[1] * 16.0 + p.v[2] * 128.0 + p.v[3] * 4096.0"
        " + p.u[0] * 16384.0 + p.u[1] * 65536.0 + Float32(p.n) * 1048576.0",
        "@fieldwise_init\nstruct Pose(Copyable, Movable):\n    var v: $vec4\n    var u: $vec2\n    var n: Int32",
    ),
}
