"""Constant-struct, constant-vector-array, constant-matrix, zero-initializer, struct-array-parameter and
unsigned-countdown probes: their C-family form (:data:`FAMILY`, as HLSL writes them) and the languages that write
them differently.

HLSL and Slang give a program-scope constant the ``static const`` qualifiers, GLSL and CrossGL ``const``, Metal
``constant``, CUDA and HIP ``__constant__`` and OpenCL C ``__constant``; WGSL and Rust declare a ``const`` and Mojo a
``comptime`` value, which a kernel copies with ``materialize`` before it reads its fields or elements at run time.
HLSL lists a matrix's scalars by row and GLSL by column, so the constant matrix lists them in the same order and HLSL
multiplies the vector on the left: every product and subscript then names the same value. CUDA, HIP, OpenCL C and
Mojo have no matrix types.

Each language writes the zero value of a struct, a vector and an array in its own shortest form: HLSL casts ``0``,
C++ value-initializes with ``{}``, C zero-fills the rest of ``{0}``, WGSL calls a constructor without arguments, Rust
uses ``Default``, ``ZERO`` and an array repeat, and GLSL, CrossGL and Mojo write the zeros out.

The unsigned countdown loop ends only when its counter wraps past zero, and the post-decrement condition leaves the
other counter at the largest unsigned value. Rust subtracts with ``wrapping_sub``, because its ``-`` traps on overflow
in a debug build.

Each probe derives its floats from the integer input, so every product and sum in it is exact: the comparison with the
native probe is bit for bit, and a driver that reassociates float arithmetic would otherwise round the two differently.
"""

from __future__ import annotations

from .base import ABSENT, PAIR, Template
from .statements import c_text

_ORIGIN_SUM = (
    "outp[i] = p.a + p.b * 16.0 + ORIGIN.a * 256.0 + ORIGIN.b * float(iin[i]) * 4096.0;"
)
_ORIGIN = f"Pair p = ORIGIN;\np.a = p.a + float(iin[i]) * 0.25;\n{_ORIGIN_SUM}"

_OFFSET_VALUES = ("0.5, 1.0", "-1.0, 2.0", "1.5, -0.5", "2.0, 3.0")
_OFFSETS = (
    "int k = iin[i] & 3;\n$vec2 o = OFFSETS[k] + OFFSETS[3 - k].yx * 0.5;\n"
    "outp[i] = o.x + o.y * 16.0 + float(iin[i]) * 64.0;"
)


def _offsets(constructor: str, suffix: str = "") -> str:
    """The four offsets, each written as ``constructor`` with ``{}`` in place of its scalars."""
    return ", ".join(
        constructor.format(", ".join(f"{x}{suffix}" for x in v.split(", ")))
        for v in _OFFSET_VALUES
    )


_ROTATE_VALUES = "0.5, -1.0, 2.0, 1.5"
_ROTATE_SUM = (
    "outp[i] = v.x + v.y * 16.0 + m[1][0] * 256.0 + ROTATE[iin[i] & 1].y * 4096.0;"
)
_ROTATE = f"$vec2 v = ROTATE * $vec2(float(iin[i]) * 0.25, float(iin[i]));\n$mat2 m = ROTATE * 2.0;\n{_ROTATE_SUM}"

_ZERO_SUM = (
    "outp[i] = p.a + p.b * 2.0 + v.x * 16.0 + v.y * 64.0 + v.w * 1024.0 + a[0] * 2048.0 + a[1] * 4096.0"
    " + a[2] * 8192.0;"
)
_ZERO_WRITES = (
    "p.b = float(iin[i]) * 0.25;\nv.y = float(iin[i]);\na[iin[i] % 3] = 2.0;\n"
)


def _zero(pair: str, vector: str, array: str) -> str:
    """The zero-initializer probe with the struct, vector and array zero values ``pair``, ``vector`` and ``array``."""
    return f"Pair p = {pair};\n$vec4 v = {vector};\nfloat a[3] = {array};\n{_ZERO_WRITES}{_ZERO_SUM}"


_PICK_CALL = (
    "Pair ps[3];\nfor (int k = 0; k < 3; k++) {\n    ps[k].a = float(k) + float(iin[i]) * 0.25;\n    ps[k].b = float(iin[i] - k);\n"
    "}\noutp[i] = pick(ps, iin[i] % 3) * 16.0 + ps[0].a;"
)
_PICK_RETURN = "    return ps[k].a + ps[k].b * 2.0 + ps[2 - k].a * 0.5;\n}"
PICK = f"float pick(Pair ps[3], int k) {{\n{_PICK_RETURN}"

_COUNTDOWN = (
    "uint n = uint(iin[i] & 7);\nfloat acc = 0.0;\nuint j = n & 3u;\nfor (; j < 4u; j--) {\n    acc += float(j) + 1.0;\n"
    "}\nuint m = n;\nwhile (m-- > 0u) {\n    acc += 16.0;\n}\n"
)
_COUNTDOWN_SUM = "outp[i] = acc + float(j >> 28u) * 1024.0 + float(m >> 30u) * 16384.0;"

FAMILY = {
    "constant-struct": Template(
        _ORIGIN, PAIR + "\n\nstatic const Pair ORIGIN = {0.5, 2.0};"
    ),
    "constant-vector-array": Template(
        _OFFSETS, f"static const $vec2 OFFSETS[4] = {{{_offsets('$vec2({})')}}};"
    ),
    "constant-matrix": Template(
        _ROTATE.replace(
            "ROTATE * $vec2(float(iin[i]) * 0.25, float(iin[i]))",
            "mul($vec2(float(iin[i]) * 0.25, float(iin[i])), ROTATE)",
        ),
        f"static const $mat2 ROTATE = $mat2({_ROTATE_VALUES});",
    ),
    "zero-initializer": Template(_zero("(Pair)0", "($vec4)0", "(float[3])0"), PAIR),
    "struct-array-parameter": Template(_PICK_CALL, f"{PAIR}\n\n{PICK}"),
    "unsigned-countdown": Template(_COUNTDOWN + _COUNTDOWN_SUM),
}

# GLSL and CrossGL write a constant with a constructor and spell out the zeros.
GLSL_ORIGIN = "const Pair ORIGIN = Pair(0.5, 2.0);"
GLSL_ROTATE = f"const $mat2 ROTATE = $mat2({_ROTATE_VALUES});"
GLSL_ZERO = _zero("Pair(0.0, 0.0)", "$vec4(0.0)", "float[3](0.0, 0.0, 0.0)")

GLSL = {
    "constant-struct": Template(_ORIGIN, f"{PAIR}\n\n{GLSL_ORIGIN}"),
    "constant-vector-array": Template(
        _OFFSETS, f"const $vec2 OFFSETS[4] = $vec2[4]({_offsets('$vec2({})')});"
    ),
    "constant-matrix": Template(_ROTATE, GLSL_ROTATE),
    "zero-initializer": Template(GLSL_ZERO, PAIR),
}

# CrossGL puts its struct and constants at shader scope, before the compute block.
CROSSGL = {
    "constant-struct": Template(_ORIGIN, prelude=f"{PAIR}\n{GLSL_ORIGIN}"),
    "constant-vector-array": Template(
        _OFFSETS,
        prelude=f"const vec2 OFFSETS[4] = {{{_offsets('vec2({})')}}};",
    ),
    "constant-matrix": Template(_ROTATE, prelude=GLSL_ROTATE),
    "zero-initializer": Template(
        _zero("Pair(0.0, 0.0)", "vec4(0.0)", "{0.0, 0.0, 0.0}"), prelude=PAIR
    ),
    "struct-array-parameter": Template(_PICK_CALL, PICK, prelude=PAIR),
}

METAL = {
    "constant-struct": Template(
        _ORIGIN, PAIR + "\n\nconstant Pair ORIGIN = {0.5, 2.0};"
    ),
    "constant-vector-array": Template(
        _OFFSETS, f"constant $vec2 OFFSETS[4] = {{{_offsets('$vec2({})')}}};"
    ),
    "constant-matrix": Template(
        _ROTATE, f"constant $mat2 ROTATE = $mat2({_ROTATE_VALUES});"
    ),
    "zero-initializer": Template(_zero("{}", "{}", "{}"), PAIR),
    "struct-array-parameter": Template(
        _PICK_CALL.replace("Pair ps[3];", "array<Pair, 3> ps;"),
        f"{PAIR}\n\nfloat pick(array<Pair, 3> ps, int k) {{\n{_PICK_RETURN}",
    ),
}

_C_PAIR = "typedef struct {\n    float a;\n    float b;\n} Pair;"


def _c(unsigned: str) -> dict[str, Template]:
    """The probes CUDA, HIP and OpenCL C share, in C with casts and ``unsigned`` for uint."""
    return {
        "constant-matrix": ABSENT,
        "unsigned-countdown": Template(c_text(_COUNTDOWN + _COUNTDOWN_SUM, unsigned)),
    }


CUDA = {
    **_c("unsigned int"),
    "constant-struct": Template(
        c_text(_ORIGIN, "unsigned int"),
        PAIR + "\n\n__constant__ Pair ORIGIN = {0.5f, 2.0f};",
    ),
    # CUDA vector types have no operators and no swizzles.
    "constant-vector-array": Template(
        "int k = iin[i] & 3;\n"
        "float2 o = make_float2(OFFSETS[k].x + OFFSETS[3 - k].y * 0.5f, OFFSETS[k].y + OFFSETS[3 - k].x * 0.5f);\n"
        "outp[i] = o.x + o.y * 16.0f + (float)(iin[i]) * 64.0f;",
        f"__constant__ float2 OFFSETS[4] = {{{_offsets('{{{}}}', 'f')}}};",
    ),
    "zero-initializer": Template(c_text(_zero("{}", "{}", "{}"), "unsigned int"), PAIR),
    "struct-array-parameter": Template(
        c_text(_PICK_CALL, "unsigned int"),
        f"{PAIR}\n\n__device__ {c_text(PICK, 'unsigned int')}",
    ),
}

OPENCL = {
    **_c("uint"),
    "constant-struct": Template(
        c_text(_ORIGIN, "uint"), _C_PAIR + "\n\n__constant Pair ORIGIN = {0.5f, 2.0f};"
    ),
    "constant-vector-array": Template(
        c_text(_OFFSETS, "uint"),
        f"__constant float2 OFFSETS[4] = {{{_offsets('(float2)({})', 'f')}}};",
    ),
    "zero-initializer": Template(
        c_text(_zero("{0}", "(float4)(0.0)", "{0}"), "uint"), _C_PAIR
    ),
    "struct-array-parameter": Template(
        c_text(_PICK_CALL, "uint"), f"{_C_PAIR}\n\n{c_text(PICK, 'uint')}"
    ),
}

_WGSL_PAIR = "struct Pair {\n    a: f32,\n    b: f32,\n}"

WGSL = {
    "constant-struct": Template(
        _ORIGIN.replace("Pair p", "var p").replace("float(", "f32("),
        f"{_WGSL_PAIR}\n\nconst ORIGIN = Pair(0.5, 2.0);",
    ),
    "constant-vector-array": Template(
        _OFFSETS.replace("int k", "let k")
        .replace("$vec2 o", "let o")
        .replace("float(", "f32("),
        f"const OFFSETS = array<vec2<f32>, 4>({_offsets('vec2<f32>({})')});",
    ),
    "constant-matrix": Template(
        _ROTATE.replace("$vec2 v", "let v")
        .replace("$mat2 m", "let m")
        .replace("float(", "f32("),
        f"const ROTATE = mat2x2<f32>({_ROTATE_VALUES});",
    ),
    "zero-initializer": Template(
        "var p = Pair();\nvar v = vec4<f32>();\nvar a = array<f32, 3>();\n"
        + (_ZERO_WRITES + _ZERO_SUM).replace("float(", "f32("),
        _WGSL_PAIR,
    ),
    "struct-array-parameter": Template(
        "var ps: array<Pair, 3>;\nfor (var k = 0; k < 3; k++) {\n    ps[k].a = f32(k) + f32(iin[i]) * 0.25;\n"
        "    ps[k].b = f32(iin[i] - k);\n}\noutp[i] = pick(ps, iin[i] % 3) * 16.0 + ps[0].a;",
        f"{_WGSL_PAIR}\n\nfn pick(ps: array<Pair, 3>, k: i32) -> f32 {{\n{_PICK_RETURN}",
    ),
    "unsigned-countdown": Template(
        "let n = u32(iin[i] & 7);\nvar acc = 0.0;\nvar j = n & 3u;\nfor (; j < 4u; j--) {\n"
        "    acc += f32(j) + 1.0;\n}\nvar m = n;\nloop {\n    let c = m;\n    m--;\n    if !(c > 0u) {\n"
        "        break;\n    }\n    acc += 16.0;\n}\n"
        + _COUNTDOWN_SUM.replace("float(", "f32(")
    ),
}

_RUST_PAIR = (
    "#[derive(Clone, Copy, Default)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}"
)

RUST = {
    "constant-struct": Template(
        "let mut p = ORIGIN;\np.a = p.a + iin[i] as f32 * 0.25;\n"
        "outp[i] = p.a + p.b * 16.0 + ORIGIN.a * 256.0 + ORIGIN.b * iin[i] as f32 * 4096.0;",
        f"{_RUST_PAIR}\n\nconst ORIGIN: Pair = Pair {{ a: 0.5, b: 2.0 }};",
    ),
    "constant-vector-array": Template(
        "let k = (iin[i] & 3) as usize;\nlet o = OFFSETS[k] + OFFSETS[3 - k].yx() * 0.5;\n"
        "outp[i] = o.x + o.y * 16.0 + iin[i] as f32 * 64.0;",
        f"const OFFSETS: [$vec2; 4] = [{_offsets('$vec2::new({})')}];",
    ),
    "constant-matrix": Template(
        "let v = ROTATE * $vec2::new(iin[i] as f32 * 0.25, iin[i] as f32);\nlet m = ROTATE * 2.0;\n"
        "outp[i] = v.x + v.y * 16.0 + m.y_axis.x * 256.0 + ROTATE.col((iin[i] & 1) as usize).y * 4096.0;",
        "const ROTATE: $mat2 = $mat2::from_cols($vec2::new(0.5, -1.0), $vec2::new(2.0, 1.5));",
    ),
    "zero-initializer": Template(
        "let mut p = Pair::default();\nlet mut v = $vec4::ZERO;\nlet mut a = [0.0f32; 3];\np.b = iin[i] as f32 * 0.25;\n"
        "v.y = iin[i] as f32;\na[(iin[i] % 3) as usize] = 2.0;\n"
        + _ZERO_SUM.removesuffix(";").replace(" + a[2]", "\n    + a[2]")
        + ";",
        _RUST_PAIR,
    ),
    "struct-array-parameter": Template(
        "let mut ps = [Pair::default(); 3];\nfor k in 0..3 {\n    ps[k].a = k as f32 + iin[i] as f32 * 0.25;\n"
        "    ps[k].b = (iin[i] - k as i32) as f32;\n}\noutp[i] = pick(ps, (iin[i] % 3) as usize) * 16.0 + ps[0].a;",
        f"{_RUST_PAIR}\n\nfn pick(ps: [Pair; 3], k: usize) -> f32 {{\n"
        "    ps[k].a + ps[k].b * 2.0 + ps[2 - k].a * 0.5\n}",
    ),
    "unsigned-countdown": Template(
        "let n = (iin[i] & 7) as u32;\nlet mut acc = 0.0;\nlet mut j = n & 3;\nwhile j < 4 {\n"
        "    acc += j as f32 + 1.0;\n    j = j.wrapping_sub(1);\n}\nlet mut m = n;\nloop {\n    let c = m;\n"
        "    m = m.wrapping_sub(1);\n    if !(c > 0) {\n        break;\n    }\n    acc += 16.0;\n}\n"
        "outp[i] = acc + (j >> 28) as f32 * 1024.0 + (m >> 30) as f32 * 16384.0;"
    ),
}

_MOJO_PAIR = "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32"

MOJO = {
    "constant-struct": Template(
        "var p = materialize[ORIGIN]()\nvar o = materialize[ORIGIN]()\np.a = p.a + Float32(iin[i]) * 0.25\n"
        "outp[i] = p.a + p.b * 16.0 + o.a * 256.0 + o.b * Float32(iin[i]) * 4096.0",
        f"{_MOJO_PAIR}\n\n\ncomptime ORIGIN = Pair(0.5, 2.0)",
    ),
    "constant-vector-array": Template(
        "var k = Int(iin[i] & 3)\nvar table = materialize[OFFSETS]()\n"
        "var o = table[k] + table[3 - k].shuffle[1, 0]() * 0.5\noutp[i] = o[0] + o[1] * 16.0 + Float32(iin[i]) * 64.0",
        f"comptime OFFSETS: Array[$vec2, 4] = [{_offsets('$vec2({})')}]",
    ),
    "constant-matrix": ABSENT,
    "zero-initializer": Template(
        "var p = Pair(0.0, 0.0)\nvar v = $vec4(0.0)\nvar a = Array[Float32, 3](fill=0.0)\np.b = Float32(iin[i]) * 0.25\n"
        "v[1] = Float32(iin[i])\na[Int(iin[i] % 3)] = 2.0\n"
        + _ZERO_SUM.removesuffix(";")
        .replace("v.x", "v[0]")
        .replace("v.y", "v[1]")
        .replace("v.w", "v[3]"),
        _MOJO_PAIR,
    ),
    "struct-array-parameter": Template(
        "var ps = Array[Pair, 3](fill=Pair(0.0, 0.0))\nfor k in range(3):\n    ps[k].a = Float32(k) + Float32(iin[i]) * 0.25\n"
        "    ps[k].b = Float32(iin[i] - Int32(k))\noutp[i] = pick(ps, Int(iin[i] % 3)) * 16.0 + ps[0].a",
        f"{_MOJO_PAIR}\n\n\ndef pick(ps: Array[Pair, 3], k: Int) -> Float32:\n"
        "    return ps[k].a + ps[k].b * 2.0 + ps[2 - k].a * 0.5",
    ),
    "unsigned-countdown": Template(
        "var n = UInt32(iin[i] & 7)\nvar acc: Float32 = 0.0\nvar j = n & 3\nwhile j < 4:\n"
        "    acc += Float32(j) + 1.0\n    j -= 1\nvar m = n\nwhile True:\n    var c = m\n    m -= 1\n"
        "    if not (c > 0):\n        break\n    acc += 16.0\n"
        "outp[i] = acc + Float32(j >> 28) * 1024.0 + Float32(m >> 30) * 16384.0"
    ),
}
