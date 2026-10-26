"""Nested-call, struct-parameter, constant, nested-loop and matrix-column probes in the languages that do not write
them as the C family does.

HLSL and Metal give a constant table the ``static const`` and ``constant`` qualifiers, CUDA and HIP put it in
``__constant__`` memory and OpenCL C in ``__constant``; CUDA and HIP mark device functions ``__device__``, and OpenCL C
converts with casts. GLSL ES 3.00 has no initializer lists. HLSL subscripts a matrix by row, but its probe lists the
scalars in the same order, so every subscript names the same value as in the CrossGL probe. WGSL parameters are
immutable, so its callee copies the struct into a ``var``. Mojo copies a struct argument only when asked to, with
``copy()``, and indexes a ``comptime`` table at run time through a ``materialize``d copy. CUDA, HIP, OpenCL C and
Mojo have no matrix types.
"""

from __future__ import annotations

from .base import ABSENT, BLEND, C_FAMILY, PAIR, SPREAD, TWICE, Template


def _f(source: str) -> str:
    """``source`` with single-precision literals."""
    return source.replace("2.0;", "2.0f;").replace("0.5;", "0.5f;")


def _c_casts(name: str) -> str:
    """The C family's probe with casts in place of the functional conversions of ints."""
    return (
        C_FAMILY[name]
        .body.replace("float(iin[i])", "(float)iin[i]")
        .replace("float(iin[i] * m)", "(float)(iin[i] * m)")
        .replace("float(j * 4 + k)", "(float)(j * 4 + k)")
        .replace("3.0", "3.0f")
    )


_TABLE = "WEIGHTS[4] = {0.125, 0.25, 0.5, 1.0};"
_C_TABLE = "WEIGHTS[4] = {0.125f, 0.25f, 0.5f, 1.0f};"
_C_PAIR = "typedef struct {\n    float a;\n    float b;\n} Pair;"

HLSL = {
    "constant-array": Template(
        C_FAMILY["constant-array"].body, f"static const float {_TABLE}"
    ),
}

METAL = {
    "constant-array": Template(
        C_FAMILY["constant-array"].body, f"constant float {_TABLE}"
    ),
}

WEBGL = {
    "constant-array": Template(
        C_FAMILY["constant-array"].body,
        "const float WEIGHTS[4] = float[4](0.125, 0.25, 0.5, 1.0);",
    ),
}

CUDA = {
    "nested-call": Template(
        C_FAMILY["nested-call"].body, f"__device__ {_f(TWICE)}\n__device__ {_f(BLEND)}"
    ),
    "struct-parameter": Template(
        C_FAMILY["struct-parameter"].body, f"{PAIR}\n__device__ {_f(SPREAD)}"
    ),
    "constant-array": Template(
        C_FAMILY["constant-array"].body, f"__constant__ float {_C_TABLE}"
    ),
    "matrix-column-assignment": ABSENT,
}

OPENCL = {
    "nested-call": Template(_c_casts("nested-call"), f"{_f(TWICE)}\n{_f(BLEND)}"),
    "struct-parameter": Template(
        _c_casts("struct-parameter"), f"{_C_PAIR}\n\n{_f(SPREAD)}"
    ),
    "const-local": Template(_c_casts("const-local")),
    "constant-array": Template(
        C_FAMILY["constant-array"].body, f"__constant float {_C_TABLE}"
    ),
    "nested-loop": Template(_c_casts("nested-loop").replace("0.0;", "0.0f;")),
    "matrix-column-assignment": ABSENT,
}

_WGSL_HELPERS = (
    "fn twice(x: f32) -> f32 {\n    return x * 2.0;\n}\n\n"
    "fn blend(x: f32, y: f32) -> f32 {\n    return x + y * 0.5;\n}"
)

WGSL = {
    "nested-call": Template(
        "outp[i] = twice(blend(fin[i], 3.0)) + blend(twice(f32(iin[i])), twice(fin[i]));",
        _WGSL_HELPERS,
    ),
    "struct-parameter": Template(
        "var p: Pair;\np.a = fin[i];\np.b = f32(iin[i]);\noutp[i] = spread(p) * 10.0 + p.a;",
        "struct Pair {\n    a: f32,\n    b: f32,\n}\n\nfn spread(q_in: Pair) -> f32 {\n    var q = q_in;\n"
        "    q.a = q.a * 2.0;\n    return q.a + q.b;\n}",
    ),
    "const-local": Template(
        "const k = 3.0;\nconst m = 2;\noutp[i] = fin[i] * k + f32(iin[i] * m);"
    ),
    "constant-array": Template(
        "let k = iin[i] & 3;\noutp[i] = WEIGHTS[k] * fin[i] + WEIGHTS[3 - k] * 16.0;",
        "const WEIGHTS = array<f32, 4>(0.125, 0.25, 0.5, 1.0);",
    ),
    "nested-loop": Template(
        "var acc: f32 = 0.0;\nfor (var j = 0; j < 4; j++) {\n    for (var k = 0; k < 4; k++) {\n"
        "        if (k > j) {\n            break;\n        }\n        if (k == (iin[i] & 3)) {\n"
        "            continue;\n        }\n        acc += f32(j * 4 + k);\n    }\n}\noutp[i] = acc + fin[i];"
    ),
    "matrix-column-assignment": Template(
        "var m = mat2x2<f32>(1.0, 2.0, 3.0, 4.0);\nm[1] = vec2<f32>(fin[i], 5.0);\nm[0].y = f32(iin[i]);\n"
        "outp[i] = m[0][0] + m[0][1] * 10.0 + m[1][0] * 100.0 + m[1][1] * 1000.0;"
    ),
}

RUST = {
    "nested-call": Template(
        "outp[i] = twice(blend(fin[i], 3.0)) + blend(twice(iin[i] as f32), twice(fin[i]));",
        "fn twice(x: f32) -> f32 {\n    x * 2.0\n}\n\nfn blend(x: f32, y: f32) -> f32 {\n    x + y * 0.5\n}",
    ),
    "struct-parameter": Template(
        "let mut p = Pair::default();\np.a = fin[i];\np.b = iin[i] as f32;\noutp[i] = spread(p) * 10.0 + p.a;",
        "#[derive(Clone, Copy, Default)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}\n\n"
        "fn spread(mut q: Pair) -> f32 {\n    q.a = q.a * 2.0;\n    q.a + q.b\n}",
    ),
    "const-local": Template(
        "const K: f32 = 3.0;\nconst M: i32 = 2;\noutp[i] = fin[i] * K + (iin[i] * M) as f32;"
    ),
    "constant-array": Template(
        "let k = (iin[i] & 3) as usize;\noutp[i] = WEIGHTS[k] * fin[i] + WEIGHTS[3 - k] * 16.0;",
        "const WEIGHTS: [f32; 4] = [0.125, 0.25, 0.5, 1.0];",
    ),
    "nested-loop": Template(
        "let mut acc = 0.0;\nfor j in 0..4 {\n    for k in 0..4 {\n        if k > j {\n            break;\n        }\n"
        "        if k == (iin[i] & 3) {\n            continue;\n        }\n        acc += (j * 4 + k) as f32;\n    }\n}\n"
        "outp[i] = acc + fin[i];"
    ),
    "matrix-column-assignment": Template(
        "let mut m = $mat2::from_cols($vec2::new(1.0, 2.0), $vec2::new(3.0, 4.0));\n"
        "m.y_axis = $vec2::new(fin[i], 5.0);\nm.x_axis.y = iin[i] as f32;\n"
        "outp[i] = m.x_axis.x + m.x_axis.y * 10.0 + m.y_axis.x * 100.0 + m.y_axis.y * 1000.0;"
    ),
}

MOJO = {
    "nested-call": Template(
        "outp[i] = twice(blend(fin[i], 3.0)) + blend(twice(Float32(iin[i])), twice(fin[i]))",
        "def twice(x: Float32) -> Float32:\n    return x * 2.0\n\n\n"
        "def blend(x: Float32, y: Float32) -> Float32:\n    return x + y * 0.5",
    ),
    "struct-parameter": Template(
        "var p = Pair(0.0, 0.0)\np.a = fin[i]\np.b = Float32(iin[i])\noutp[i] = spread(p.copy()) * 10.0 + p.a",
        "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32\n\n\n"
        "def spread(var q: Pair) -> Float32:\n    q.a = q.a * 2.0\n    return q.a + q.b",
    ),
    "const-local": Template(
        "comptime k: Float32 = 3.0\ncomptime m: Int32 = 2\noutp[i] = fin[i] * k + Float32(iin[i] * m)"
    ),
    "constant-array": Template(
        "var k = Int(iin[i] & 3)\nvar table = materialize[WEIGHTS]()\n"
        "outp[i] = table[k] * fin[i] + table[3 - k] * 16.0",
        "comptime WEIGHTS: Array[Float32, 4] = [0.125, 0.25, 0.5, 1.0]",
    ),
    "nested-loop": Template(
        "var acc: Float32 = 0.0\nfor j in range(4):\n    for k in range(4):\n        if k > j:\n            break\n"
        "        if k == Int(iin[i] & 3):\n            continue\n        acc += Float32(j * 4 + k)\n"
        "outp[i] = acc + fin[i]"
    ),
    "matrix-column-assignment": ABSENT,
}
