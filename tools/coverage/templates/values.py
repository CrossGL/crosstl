"""Value-parameter-mutation, matrix-parameter, struct-matrix-member, struct-conditional, global-array and
color-swizzle probes: their C-family form (:data:`FAMILY`) and the languages that write them differently.

The matrices are symmetric and the probes write only their diagonals, so they read the same whether a language fills
and indexes a matrix by columns, as GLSL does, or by rows, as HLSL does. WGSL and Rust parameters are immutable, so
the WGSL callee copies them into ``var``s and the Rust one marks them ``mut``; Mojo takes them as ``var`` arguments.
HLSL's conditional operator and WGSL's ``select`` take no structs, so their struct-conditional probes branch, and
WGSL cannot assign to a swizzle, so its color-swizzle probe writes one component at a time. HLSL marks a private
global ``static`` and WGSL declares it ``var<private>``. CUDA, HIP, OpenCL C and Mojo have no matrices; CUDA, HIP,
Rust and Mojo name vector components x, y, z and w only; and Metal, CUDA, HIP, OpenCL C, Rust and Mojo have no global
that each invocation holds a copy of.
"""

from __future__ import annotations

from .base import ABSENT, PAIR, Template
from .statements import c_text

# twist changes all three of its by-value parameters, counting n down as a loop counter; the caller's stay as they were.
TWIST = (
    "float twist(float x, $vec2 v, int n) {\n    x = x * 2.0;\n    v.y = v.y + x;\n    float acc = 0.0;\n"
    "    while (n > 0) {\n        acc += v.y;\n        n -= 2;\n    }\n"
    "    return x + v.x * 4.0 + acc * 16.0 + float(n);\n}"
)
# bend changes its matrix parameter's diagonal before it scales it.
BEND = (
    "$mat2 bend($mat2 m, float s) {\n    m[1][1] = m[1][1] + s;\n    return m * s;\n}"
)
BASIS = "struct Basis {\n    $mat2 m;\n    float w;\n};"
# stash adds to an element of the invocation's own copy of table.
STASH = (
    "float table[4];\n\nvoid stash(int k, float v) {\n    table[k] = table[k] + v;\n}"
)

_TWIST_SUM = "r + (x + v.x * 2.0 + v.y * 8.0 + float(n) * 32.0) * 256.0"
_BEND_SUM = "n{c0}{r0} + n{c0}{r1} * 16.0 + n{c1}{r1} * 256.0 + m{c1}{r1} * 4096.0"
_BASIS_SUM = "b.m{c0}{r0} + b.m{c0}{r1} * 16.0 + b.m{c1}{r1} * 256.0 + c.m{c0}{r0} * 4096.0 + c.m{c1}{r1} * 8192.0"
_PICK_SUM = "r.a + r.b * 16.0 + s.a * 256.0 + s.b * 4096.0 + q.b * 65536.0"
_TABLE_SUM = "table[0] + table[1] * 16.0 + table[2] * 256.0 + table[3] * 4096.0"
_COLOR_SUM = "c.r + c.g * 16.0 + c.b * 256.0 + c.a * 4096.0 + t.g * 65536.0"
_INDEXED = {"c0": "[0]", "c1": "[1]", "r0": "[0]", "r1": "[1]"}
_AXES = {"c0": ".x_axis", "c1": ".y_axis", "r0": ".x", "r1": ".y"}

FAMILY = {
    "value-parameter-mutation": Template(
        "float x = fin[i];\n$vec2 v = $vec2(1.0, 0.5);\nint n = iin[i] & 7;\nfloat r = twist(x, v, n);\n"
        f"outp[i] = {_TWIST_SUM};",
        TWIST,
    ),
    "matrix-parameter": Template(
        "float f = float(iin[i]);\n$mat2 m = $mat2(f, 1.0, 1.0, 3.0);\n$mat2 n = bend(m, 2.0);\n"
        f"outp[i] = {_BEND_SUM.format(**_INDEXED)};",
        BEND,
    ),
    # Assigning the struct copies its matrix, so the write to c.m[0][0] must not reach b.
    "struct-matrix-member": Template(
        "float f = float(iin[i]);\nBasis b;\nb.m = $mat2(f, 2.0, 2.0, 1.0);\nb.w = 3.0;\n"
        "b.m[1][1] = b.m[0][0] + b.w;\nb.m = b.m * 2.0;\nBasis c = b;\nc.m[0][0] = 0.5;\n"
        f"outp[i] = {_BASIS_SUM.format(**_INDEXED)};",
        BASIS,
    ),
    # A conditional expression picks a whole struct; the write to s.b changes the copy only.
    "struct-conditional": Template(
        "Pair p;\np.a = fin[i];\np.b = 1.0;\nPair q;\nq.a = 2.0;\nq.b = float(iin[i]);\n"
        "Pair r = iin[i] > 7 ? p : q;\nPair s = (iin[i] & 1) == 0 ? q : p;\ns.b = s.b + 4.0;\n"
        f"outp[i] = {_PICK_SUM};",
        PAIR,
    ),
    "global-array": Template(
        "for (int k = 0; k < 4; k++) {\n    table[k] = float(k);\n}\nstash(iin[i] & 3, fin[i]);\nstash(3, 0.5);\n"
        f"outp[i] = {_TABLE_SUM};",
        STASH,
    ),
    # The rgba names of the components: a three-component read, a two-component write and single components.
    "color-swizzle": Template(
        "$vec4 c = $vec4(fin[i], 1.0, 2.0, 3.0);\n$vec3 t = c.bgr;\nc.ga = $vec2(float(iin[i]), 5.0);\n"
        f"c.r = c.r + t.r;\noutp[i] = {_COLOR_SUM};"
    ),
}

HLSL = {
    "struct-conditional": Template(
        "Pair p;\np.a = fin[i];\np.b = 1.0;\nPair q;\nq.a = 2.0;\nq.b = float(iin[i]);\nPair r = q;\n"
        "if (iin[i] > 7) {\n    r = p;\n}\nPair s = p;\nif ((iin[i] & 1) == 0) {\n    s = q;\n}\n"
        f"s.b = s.b + 4.0;\noutp[i] = {_PICK_SUM};",
        PAIR,
    ),
    "global-array": Template(FAMILY["global-array"].body, "static " + STASH),
}

METAL = {"global-array": ABSENT}


def _c(name: str, unsigned: str, vector: str) -> str:
    """The C family's probe with casts, single-precision literals and ``vector`` constructing a float2."""
    return c_text(FAMILY[name].body, unsigned).replace("$vec2(", vector)


_C_TWIST = c_text(TWIST, "uint").replace("$vec2", "float2")
_NO_MATRICES = {"matrix-parameter": ABSENT, "struct-matrix-member": ABSENT}

CUDA = {
    "value-parameter-mutation": Template(
        _c("value-parameter-mutation", "unsigned int", "make_float2("),
        "__device__ " + _C_TWIST,
    ),
    **_NO_MATRICES,
    "struct-conditional": Template(
        c_text(FAMILY["struct-conditional"].body, "unsigned int"), PAIR
    ),
    "global-array": ABSENT,
    "color-swizzle": ABSENT,
}

OPENCL = {
    "value-parameter-mutation": Template(
        _c("value-parameter-mutation", "uint", "(float2)("), _C_TWIST
    ),
    **_NO_MATRICES,
    "struct-conditional": Template(
        c_text(FAMILY["struct-conditional"].body, "uint"),
        "typedef struct {\n    float a;\n    float b;\n} Pair;",
    ),
    "global-array": ABSENT,
    "color-swizzle": Template(
        "float4 c = (float4)(fin[i], 1.0f, 2.0f, 3.0f);\nfloat3 t = c.bgr;\nc.ga = (float2)((float)iin[i], 5.0f);\n"
        "c.r = c.r + t.r;\noutp[i] = " + c_text(_COLOR_SUM, "uint") + ";"
    ),
}

_WGSL_PAIR = "struct Pair {\n    a: f32,\n    b: f32,\n}"

WGSL = {
    "value-parameter-mutation": Template(
        "let x = fin[i];\nlet v = vec2<f32>(1.0, 0.5);\nlet n = iin[i] & 7;\nlet r = twist(x, v, n);\n"
        f"outp[i] = {_TWIST_SUM.replace('float(', 'f32(')};",
        "fn twist(x_in: f32, v_in: vec2<f32>, n_in: i32) -> f32 {\n    var x = x_in;\n    var v = v_in;\n"
        "    var n = n_in;\n    x = x * 2.0;\n    v.y = v.y + x;\n    var acc: f32 = 0.0;\n    while (n > 0) {\n"
        "        acc += v.y;\n        n -= 2;\n    }\n    return x + v.x * 4.0 + acc * 16.0 + f32(n);\n}",
    ),
    "matrix-parameter": Template(
        "let f = f32(iin[i]);\nlet m = mat2x2<f32>(f, 1.0, 1.0, 3.0);\nlet n = bend(m, 2.0);\n"
        f"outp[i] = {_BEND_SUM.format(**_INDEXED)};",
        "fn bend(m_in: mat2x2<f32>, s: f32) -> mat2x2<f32> {\n    var m = m_in;\n    m[1][1] = m[1][1] + s;\n"
        "    return m * s;\n}",
    ),
    "struct-matrix-member": Template(
        "let f = f32(iin[i]);\nvar b: Basis;\nb.m = mat2x2<f32>(f, 2.0, 2.0, 1.0);\nb.w = 3.0;\n"
        "b.m[1][1] = b.m[0][0] + b.w;\nb.m = b.m * 2.0;\nvar c = b;\nc.m[0][0] = 0.5;\n"
        f"outp[i] = {_BASIS_SUM.format(**_INDEXED)};",
        "struct Basis {\n    m: mat2x2<f32>,\n    w: f32,\n}",
    ),
    "struct-conditional": Template(
        "var p: Pair;\np.a = fin[i];\np.b = 1.0;\nvar q: Pair;\nq.a = 2.0;\nq.b = f32(iin[i]);\nvar r = q;\n"
        "if (iin[i] > 7) {\n    r = p;\n}\nvar s = p;\nif ((iin[i] & 1) == 0) {\n    s = q;\n}\n"
        f"s.b = s.b + 4.0;\noutp[i] = {_PICK_SUM};",
        _WGSL_PAIR,
    ),
    "global-array": Template(
        "for (var k = 0; k < 4; k++) {\n    table[k] = f32(k);\n}\nstash(iin[i] & 3, fin[i]);\nstash(3, 0.5);\n"
        f"outp[i] = {_TABLE_SUM};",
        "var<private> table: array<f32, 4>;\n\nfn stash(k: i32, v: f32) {\n    table[k] = table[k] + v;\n}",
    ),
    "color-swizzle": Template(
        "var c = vec4<f32>(fin[i], 1.0, 2.0, 3.0);\nlet t = c.bgr;\nc.g = f32(iin[i]);\nc.a = 5.0;\n"
        f"c.r = c.r + t.r;\noutp[i] = {_COLOR_SUM};"
    ),
}

RUST = {
    "value-parameter-mutation": Template(
        "let x = fin[i];\nlet v = $vec2::new(1.0, 0.5);\nlet n = iin[i] & 7;\nlet r = twist(x, v, n);\n"
        "outp[i] = r + (x + v.x * 2.0 + v.y * 8.0 + n as f32 * 32.0) * 256.0;",
        "fn twist(mut x: f32, mut v: $vec2, mut n: i32) -> f32 {\n    x = x * 2.0;\n    v.y = v.y + x;\n"
        "    let mut acc = 0.0;\n    while n > 0 {\n        acc += v.y;\n        n -= 2;\n    }\n"
        "    x + v.x * 4.0 + acc * 16.0 + n as f32\n}",
    ),
    "matrix-parameter": Template(
        "let f = iin[i] as f32;\nlet m = $mat2::from_cols($vec2::new(f, 1.0), $vec2::new(1.0, 3.0));\n"
        f"let n = bend(m, 2.0);\noutp[i] = {_BEND_SUM.format(**_AXES)};",
        "fn bend(mut m: $mat2, s: f32) -> $mat2 {\n    m.y_axis.y = m.y_axis.y + s;\n    m * s\n}",
    ),
    "struct-matrix-member": Template(
        "let f = iin[i] as f32;\n"
        "let mut b = Basis { m: $mat2::from_cols($vec2::new(f, 2.0), $vec2::new(2.0, 1.0)), w: 3.0 };\n"
        "b.m.y_axis.y = b.m.x_axis.x + b.w;\nb.m = b.m * 2.0;\nlet mut c = b;\nc.m.x_axis.x = 0.5;\n"
        f"outp[i] = {_BASIS_SUM.format(**_AXES)};",
        "#[derive(Clone, Copy)]\nstruct Basis {\n    m: $mat2,\n    w: f32,\n}",
    ),
    "struct-conditional": Template(
        "let p = Pair { a: fin[i], b: 1.0 };\nlet q = Pair { a: 2.0, b: iin[i] as f32 };\n"
        "let r = if iin[i] > 7 { p } else { q };\nlet mut s = if (iin[i] & 1) == 0 { q } else { p };\n"
        f"s.b = s.b + 4.0;\noutp[i] = {_PICK_SUM};",
        "#[derive(Clone, Copy, Default)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}",
    ),
    "global-array": ABSENT,
    "color-swizzle": ABSENT,
}

MOJO = {
    "value-parameter-mutation": Template(
        "var x = fin[i]\nvar v = $vec2(1.0, 0.5)\nvar n = iin[i] & 7\nvar r = twist(x, v, n)\n"
        "outp[i] = r + (x + v[0] * 2.0 + v[1] * 8.0 + Float32(n) * 32.0) * 256.0",
        "def twist(var x: Float32, var v: $vec2, var n: Int32) -> Float32:\n    x = x * 2.0\n    v[1] = v[1] + x\n"
        "    var acc: Float32 = 0.0\n    while n > 0:\n        acc += v[1]\n        n -= 2\n"
        "    return x + v[0] * 4.0 + acc * 16.0 + Float32(n)",
    ),
    **_NO_MATRICES,
    "struct-conditional": Template(
        "var p = Pair(fin[i], 1.0)\nvar q = Pair(2.0, Float32(iin[i]))\n"
        "var r = p.copy() if iin[i] > 7 else q.copy()\nvar s = q.copy() if (iin[i] & 1) == 0 else p.copy()\n"
        f"s.b = s.b + 4.0\noutp[i] = {_PICK_SUM}",
        "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32",
    ),
    "global-array": ABSENT,
    "color-swizzle": ABSENT,
}
