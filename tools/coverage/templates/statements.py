"""Unsigned-float-conversion, chained-assignment, switch-default-position, for-loop-variants, void-return and
expression-swizzle probes in the languages that do not write them as the C family does.

Metal passes references as ``thread`` references. CUDA and HIP pass references, mark device functions ``__device__``
and have no swizzles; OpenCL C passes pointers, converts with casts and builds vectors with vector literals. WGSL
passes pointers, lists the labels of a case together, and has no assignment expressions, so it chains assignments
as separate statements, as Rust and Mojo do. Rust matches with ``match``, whose catch-all arm comes last, and loops
without the C family's ``for`` with ``while``. Mojo has no ``switch``, loops with ``while`` and swizzles with
``shuffle``.
"""

from __future__ import annotations

import re

from .base import ABSENT, C_FAMILY, LIMIT, Template

_LITERAL = re.compile(r"\b(\d+\.\d+)\b")
_SWIZZLE_SUM = (
    "a.x + a.y * 16.0 + b.x * 256.0 + b.y * 4096.0 + d.x * 65536.0 + d.y * 1048576.0"
)
_CHAINED_SUM = "a + b * 32.0 + c * 1024.0"
_LOOP_SUM = "acc + {0}(k) * 16384.0 + {0}(j) * 262144.0"


def _c(name: str, unsigned: str) -> str:
    """The C family's probe with casts in place of conversions, ``unsigned`` for uint and single-precision literals."""
    body = (
        C_FAMILY[name]
        .body.replace("float(", "(float)(")
        .replace("uint(", f"({unsigned})(")
    )
    return _LITERAL.sub(r"\1f", re.sub(r"\buint\b", unsigned, body))


def _c_vectors(name: str, unsigned: str, construct: str) -> str:
    """:func:`_c` with ``float2`` and ``float4`` for the vector types, built with ``construct``."""
    body = _c(name, unsigned)
    for width in ("2", "4"):
        body = body.replace(f"$vec{width}(", construct.format(width)).replace(
            f"$vec{width}", f"float{width}"
        )
    return body


METAL = {
    "void-return": Template(
        C_FAMILY["void-return"].body, LIMIT.replace("inout float x", "thread float& x")
    ),
}

CUDA = {
    "unsigned-float-conversion": Template(
        _c("unsigned-float-conversion", "unsigned int")
    ),
    "chained-assignment": Template(
        _c_vectors("chained-assignment", "unsigned int", "make_float{}(")
    ),
    "switch-default-position": Template(_c("switch-default-position", "unsigned int")),
    "for-loop-variants": Template(_c("for-loop-variants", "unsigned int")),
    "void-return": Template(
        _c("void-return", "unsigned int"),
        "__device__ "
        + _LITERAL.sub(r"\1f", LIMIT.replace("inout float x", "float& x")),
    ),
    "expression-swizzle": ABSENT,
}

OPENCL = {
    "unsigned-float-conversion": Template(_c("unsigned-float-conversion", "uint")),
    "chained-assignment": Template(
        _c_vectors("chained-assignment", "uint", "(float{})(")
    ),
    "switch-default-position": Template(_c("switch-default-position", "uint")),
    "for-loop-variants": Template(_c("for-loop-variants", "uint")),
    "void-return": Template(
        _c("void-return", "uint")
        .replace("limit(x", "limit(&x")
        .replace("limit(y", "limit(&y"),
        "void limit(float* x, float hi) {\n    if (*x > hi) {\n        *x = hi;\n        return;\n    }\n"
        "    *x = *x * 2.0f;\n}",
    ),
    "expression-swizzle": Template(
        _c_vectors("expression-swizzle", "uint", "(float{})(")
    ),
}

WGSL = {
    "unsigned-float-conversion": Template(
        "let k = u32(iin[i] & 15);\nlet big = f32(k) * 268435456.0;\nlet u = u32(big);\nlet v = u + 2147483648u;\n"
        "let back = f32(v);\noutp[i] = f32(u >> 24u) + f32(v >> 28u) * 512.0 + f32(u32(f32(k) * 2.75)) * 8192.0"
        " + select(0.0, 524288.0, back >= 2147483648.0);"
    ),
    "chained-assignment": Template(
        "var c = f32(iin[i]) * 2.0;\nvar b = c;\nlet a = b;\nlet k = iin[i] + 1;\nlet j = k;\n"
        "let w = vec2<f32>(1.0, f32(j));\nlet v = w;\nb = c + 1.0;\nc = b * 2.0;\n"
        f"outp[i] = {_CHAINED_SUM} + f32(k) * 65536.0 + select(0.0, 2097152.0, v.y == w.y);"
    ),
    "switch-default-position": Template(
        "let k = iin[i] - 3;\nvar acc: f32 = 0.0;\nswitch (k) {\n    default: {\n        acc = 1.0;\n    }\n"
        "    case -1: {\n        acc = 2.0;\n    }\n    case 2, 5: {\n        acc = 4.0;\n    }\n"
        "    case 0: {\n        acc = 8.0;\n    }\n}\noutp[i] = acc + f32(k) * 16.0;"
    ),
    "for-loop-variants": Template(
        "var k = iin[i] & 7;\nvar acc: f32 = 0.0;\nfor (; k < 10;) {\n    acc += 1.0;\n    k += 3;\n}\nvar j = 0;\n"
        "for (; j < 4; j++) {\n    acc += 16.0;\n}\nfor (var u = 4u; u > 0u; u--) {\n    acc += f32(u) * 64.0;\n}\n"
        "for (var m = 9; m >= 0; m -= 4) {\n    acc += 1024.0;\n}\n"
        f"outp[i] = {_LOOP_SUM.format('f32')};"
    ),
    "void-return": Template(
        "var x = f32(iin[i]);\nlimit(&x, 9.0);\nvar y = f32(iin[i] & 3);\nlimit(&y, 2.0);\noutp[i] = x + y * 32.0;",
        "fn limit(x: ptr<function, f32>, hi: f32) {\n    if (*x > hi) {\n        *x = hi;\n        return;\n    }\n"
        "    *x = *x * 2.0;\n}",
    ),
    "expression-swizzle": Template(
        "let v = vec4<f32>(f32(iin[i]), 1.0, 2.0, 3.0);\nlet a = (v * 2.0).wy;\nlet b = (v + v.wzyx).zx;\n"
        "var arr: array<vec4<f32>, 2>;\narr[0] = v;\narr[1] = v.wzyx;\nlet d = arr[iin[i] & 1].zx;\n"
        f"outp[i] = {_SWIZZLE_SUM};"
    ),
}

RUST = {
    "unsigned-float-conversion": Template(
        "let k = (iin[i] & 15) as u32;\nlet big = k as f32 * 268435456.0;\nlet u = big as u32;\n"
        "let v = u.wrapping_add(2147483648);\nlet back = v as f32;\n"
        "let w = if back >= 2147483648.0 { 524288.0 } else { 0.0 };\n"
        "outp[i] = (u >> 24) as f32 + (v >> 28) as f32 * 512.0 + ((k as f32 * 2.75) as u32) as f32 * 8192.0 + w;"
    ),
    "chained-assignment": Template(
        "let mut c = iin[i] as f32 * 2.0;\nlet mut b = c;\nlet a = b;\nlet k = iin[i] + 1;\nlet j = k;\n"
        "let w = $vec2::new(1.0, j as f32);\nlet v = w;\nb = c + 1.0;\nc = b * 2.0;\n"
        "let e = if v.y == w.y { 2097152.0 } else { 0.0 };\n"
        f"outp[i] = {_CHAINED_SUM} + k as f32 * 65536.0 + e;"
    ),
    "switch-default-position": Template(
        "let k = iin[i] - 3;\nlet acc = match k {\n    -1 => 2.0,\n    2 | 5 => 4.0,\n    0 => 8.0,\n    _ => 1.0,\n};\n"
        "outp[i] = acc + k as f32 * 16.0;"
    ),
    "for-loop-variants": Template(
        "let mut k = iin[i] & 7;\nlet mut acc = 0.0;\nwhile k < 10 {\n    acc += 1.0;\n    k += 3;\n}\n"
        "let mut j = 0;\nwhile j < 4 {\n    acc += 16.0;\n    j += 1;\n}\nlet mut u = 4u32;\nwhile u > 0 {\n"
        "    acc += u as f32 * 64.0;\n    u -= 1;\n}\nlet mut m = 9;\nwhile m >= 0 {\n    acc += 1024.0;\n    m -= 4;\n}\n"
        "outp[i] = acc + k as f32 * 16384.0 + j as f32 * 262144.0;"
    ),
    "void-return": Template(
        "let mut x = iin[i] as f32;\nlimit(&mut x, 9.0);\nlet mut y = (iin[i] & 3) as f32;\nlimit(&mut y, 2.0);\n"
        "outp[i] = x + y * 32.0;",
        "fn limit(x: &mut f32, hi: f32) {\n    if *x > hi {\n        *x = hi;\n        return;\n    }\n"
        "    *x = *x * 2.0;\n}",
    ),
    "expression-swizzle": Template(
        "let v = $vec4::new(iin[i] as f32, 1.0, 2.0, 3.0);\nlet a = (v * 2.0).wy();\nlet b = (v + v.wzyx()).zx();\n"
        "let arr = [v, v.wzyx()];\nlet d = arr[(iin[i] & 1) as usize].zx();\n"
        f"outp[i] = {_SWIZZLE_SUM};"
    ),
}

MOJO = {
    "unsigned-float-conversion": Template(
        "var k = UInt32(iin[i] & 15)\nvar big = Float32(k) * 268435456.0\nvar u = UInt32(big)\n"
        "var v = u + 2147483648\nvar back = Float32(v)\nvar w: Float32 = 524288.0 if back >= 2147483648.0 else 0.0\n"
        "outp[i] = Float32(u >> 24) + Float32(v >> 28) * 512.0 + Float32(UInt32(Float32(k) * 2.75)) * 8192.0 + w"
    ),
    "chained-assignment": Template(
        "var c = Float32(iin[i]) * 2.0\nvar b = c\nvar a = b\nvar k = iin[i] + 1\nvar j = k\n"
        "var w = $vec2(1.0, Float32(j))\nvar v = w\nb = c + 1.0\nc = b * 2.0\n"
        "var e: Float32 = 2097152.0 if v[1] == w[1] else 0.0\n"
        f"outp[i] = {_CHAINED_SUM} + Float32(k) * 65536.0 + e"
    ),
    "switch-default-position": ABSENT,
    "for-loop-variants": Template(
        "var k = iin[i] & 7\nvar acc: Float32 = 0.0\nwhile k < 10:\n    acc += 1.0\n    k += 3\nvar j = 0\n"
        "while j < 4:\n    acc += 16.0\n    j += 1\nvar u: UInt32 = 4\nwhile u > 0:\n    acc += Float32(u) * 64.0\n"
        "    u -= 1\nvar m = 9\nwhile m >= 0:\n    acc += 1024.0\n    m -= 4\n"
        f"outp[i] = {_LOOP_SUM.format('Float32')}"
    ),
    "void-return": Template(
        "var x = Float32(iin[i])\nlimit(x, 9.0)\nvar y = Float32(iin[i] & 3)\nlimit(y, 2.0)\noutp[i] = x + y * 32.0",
        "def limit(mut x: Float32, hi: Float32):\n    if x > hi:\n        x = hi\n        return\n    x = x * 2.0",
    ),
    "expression-swizzle": Template(
        "var v = $vec4(Float32(iin[i]), 1.0, 2.0, 3.0)\nvar a = (v * 2.0).shuffle[3, 1]()\n"
        "var b = (v + v.shuffle[3, 2, 1, 0]()).shuffle[2, 0]()\nvar arr = Array[$vec4, 2](fill=v)\n"
        "arr[1] = v.shuffle[3, 2, 1, 0]()\nvar d = arr[Int(iin[i] & 1)].shuffle[2, 0]()\n"
        "outp[i] = a[0] + a[1] * 16.0 + b[0] * 256.0 + b[1] * 4096.0 + d[0] * 65536.0 + d[1] * 1048576.0"
    ),
}
