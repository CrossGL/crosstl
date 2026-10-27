"""Octal-literal, literal-suffixes, unsigned-negation, call-result-access, reserved-identifier and
sibling-redeclaration probes: their C-family form (:data:`FAMILY`) and the languages that write them differently.

CrossGL writes an octal literal as ``0o17``, as Rust and Mojo do; the C family writes ``017``, and WGSL has no octal
literals. GLSL ES 3.00 has no float suffix, and Mojo no literal suffix at all. WGSL has no unary minus on ``u32``
(WGSL, section 8.7), so its probe subtracts from ``0u``, and Rust's probe calls ``wrapping_neg``. CUDA's vectors have
no swizzles, so its probe reads the components one by one.

Each reserved-identifier probe names its helper, the helper's parameter and five locals with words that its own
language allows and others reserve: ``sample``, ``input``, ``output``, ``filter`` and ``half`` in GLSL; ``half`` in
HLSL; ``let`` and ``var`` in Slang; ``device``, ``kernel``, ``thread`` and ``half`` in Metal; ``fn``, ``let``,
``type`` and ``mod`` in WGSL and Rust; ``fn`` and ``var`` in Mojo.
"""

from __future__ import annotations

from .base import ABSENT, BINS, MAKE_PAIR, PAIR, Template
from .statements import c_text

_OCTAL = (
    "int a = 017 + 00;\nuint b = 010u + uint(iin[i]);\nint c = (iin[i] & 07) * 011;\n"
    "outp[i] = float(a) + float(b) * 32.0 + float(c) * 1024.0;"
)
_SUFFIXES = (
    "float a = 0.5f + 3. + 2e+1 + 1.5F;\nuint b = 7U + uint(iin[i]);\nfloat c = float(iin[i]) * 25e-2f;\n"
    "outp[i] = a + float(b) * 32.0 + c * 1024.0;"
)
# Every result is a power-of-two multiple of a whole number, so each sum is exact.
_NEGATION_SUM = "{0}(b >> 28u) + {0}(c >> 28u) * 16.0 + {0}(d == b ? 1 : 0) * 256.0 + {0}(c & 15u) * 512.0"
_NEGATION = (
    "uint a = uint(iin[i]);\nuint b = -a;\nuint c = -(a + 1u) + 5u;\nuint d = ~a + 1u;\n"
    f"outp[i] = {_NEGATION_SUM.format('float')};"
)
_ACCESS_SUM = "a + s{x} * 32.0 + s{y} * 1024.0 + c * 32768.0 + d * 1048576.0"
_ACCESS = (
    "float x = float(iin[i] & 7);\nfloat a = make(x).b;\n$vec2 s = spread4(x).wy;\nfloat c = fill(x).v[iin[i] % 3];\n"
    "float d = make(spread4(x).z).a + float(fill(x).n);\n"
    f"outp[i] = {_ACCESS_SUM.format(x='.x', y='.y')};"
)
FILL = (
    "Bins fill(float x) {\n    Bins b;\n    for (int k = 0; k < 3; k++) {\n        b.v[k] = x + float(k);\n    }\n"
    "    b.n = 2;\n    return b;\n}"
)
SPREAD4 = "$vec4 spread4(float x) {\n    return $vec4(x, x + 1.0, x + 2.0, x + 3.0);\n}"
# Two blocks, two loops and the two branches of an if each declare a name their sibling declares with another type.
_SIBLINGS = (
    "float r = 0.0;\n{\n    int t = iin[i] + 1;\n    r += float(t);\n}\n{\n    float t = float(iin[i]) * 0.5;\n"
    "    r += t * 32.0;\n}\nfor (int k = 0; k < 2; k++) {\n    r += float(k) * 512.0;\n}\n"
    "for (uint k = 2u; k < 4u; k++) {\n    r += float(k) * 1024.0;\n}\nif (iin[i] > 7) {\n"
    "    $vec2 t = $vec2(1.0, 2.0);\n    r += t.y * 8192.0;\n} else {\n    bool t = true;\n"
    "    r += t ? 32768.0 : 0.0;\n}\noutp[i] = r;"
)

CROSSGL_NAMES = ("sample", "type", "device", "let", "half", "fn", "output")


def reserved(names: tuple[str, ...]) -> Template:
    """The reserved-identifier probe in C-family syntax with ``names``: the helper, its parameter, then five locals."""
    f, p, a, b, c, d, e = names
    return Template(
        f"float {a} = float(iin[i]);\nint {b} = iin[i] & 7;\nfloat {c} = {f}({a});\nint {d} = {b} * 3 + 1;\n"
        f"float {e} = {c} + float({d}) * 32.0;\noutp[i] = {e};",
        f"float {f}(float {p}) {{\n    return {p} * 2.0;\n}}",
    )


FAMILY = {
    "octal-literal": Template(_OCTAL),
    "literal-suffixes": Template(_SUFFIXES),
    "unsigned-negation": Template(_NEGATION),
    "call-result-access": Template(
        _ACCESS, f"{PAIR}\n{BINS}\n{MAKE_PAIR}\n\n{FILL}\n\n{SPREAD4}"
    ),
    "reserved-identifier": reserved(
        ("loop", "type", "device", "kernel", "thread", "mod", "match")
    ),
    "sibling-redeclaration": Template(_SIBLINGS),
}

METAL = {
    "reserved-identifier": reserved(
        ("sample", "input", "output", "let", "var", "mod", "filter")
    ),
}

WEBGL = {
    "literal-suffixes": Template(
        _SUFFIXES.replace("0.5f", "0.5")
        .replace("1.5F", "1.5")
        .replace("25e-2f", "25e-2")
    ),
}


def _c_family(unsigned: str, device: str, vector: str) -> dict[str, Template]:
    """The C family's probes as CUDA, HIP (``device`` ``__device__ ``) and OpenCL C (``device`` empty) write them.

    ``vector`` builds a ``float4`` from its four components: ``make_float4(`` in CUDA, ``(float4)(`` in OpenCL C.
    """
    pair = (
        "struct Pair {\n    float a;\n    float b;\n};"
        if device
        else ("typedef struct {\n    float a;\n    float b;\n} Pair;")
    )
    bins = (
        "struct Bins {\n    float v[3];\n    int n;\n};"
        if device
        else ("typedef struct {\n    float v[3];\n    int n;\n} Bins;")
    )
    helpers = "\n\n".join(
        device + c_text(text, unsigned)
        for text in (
            MAKE_PAIR,
            FILL,
            SPREAD4.replace("$vec4(", vector).replace("$vec4", "float4"),
        )
    )
    if device:
        access = c_text(_ACCESS, unsigned).replace(
            "$vec2 s = spread4(x).wy;",
            "float sx = spread4(x).w;\nfloat sy = spread4(x).y;",
        )
        access = access.replace("s.x *", "sx *").replace("s.y *", "sy *")
    else:
        access = c_text(_ACCESS, unsigned).replace("$vec2", "float2")
    two = "make_float2(" if device else "(float2)("
    names = reserved(("fn", "type", "input", "let", "var", "mod", "output"))
    return {
        "octal-literal": Template(c_text(_OCTAL, unsigned)),
        "literal-suffixes": Template(
            c_text(_SUFFIXES, unsigned)
            .replace("3. +", "3.f +")
            .replace("2e+1 +", "2e+1f +")
        ),
        "unsigned-negation": Template(c_text(_NEGATION, unsigned)),
        "call-result-access": Template(access, f"{pair}\n{bins}\n\n{helpers}"),
        "reserved-identifier": Template(
            c_text(names.body, unsigned), device + c_text(names.declarations, unsigned)
        ),
        "sibling-redeclaration": Template(
            c_text(_SIBLINGS, unsigned)
            .replace("$vec2(", two)
            .replace("$vec2", "float2")
        ),
    }


CUDA = _c_family("unsigned int", "__device__ ", "make_float4(")
OPENCL = _c_family("uint", "", "(float4)(")

WGSL = {
    "octal-literal": ABSENT,
    "literal-suffixes": Template(
        "let a = 0.5f + 3. + 2e+1 + 1.5f;\nlet b = 7u + u32(iin[i]);\nlet c = f32(iin[i]) * 25e-2f;\n"
        "outp[i] = a + f32(b) * 32.0 + c * 1024.0;"
    ),
    "unsigned-negation": Template(
        "let a = u32(iin[i]);\nlet b = 0u - a;\nlet c = (0u - (a + 1u)) + 5u;\nlet d = ~a + 1u;\n"
        "outp[i] = f32(b >> 28u) + f32(c >> 28u) * 16.0 + f32(select(0, 1, d == b)) * 256.0"
        " + f32(c & 15u) * 512.0;"
    ),
    "call-result-access": Template(
        "let x = f32(iin[i] & 7);\nlet a = make(x).b;\nlet s = spread4(x).wy;\nlet c = fill(x).v[iin[i] % 3];\n"
        "let d = make(spread4(x).z).a + f32(fill(x).n);\n"
        f"outp[i] = {_ACCESS_SUM.format(x='.x', y='.y')};",
        "struct Pair {\n    a: f32,\n    b: f32,\n}\n\nstruct Bins {\n    v: array<f32, 3>,\n    n: i32,\n}\n\n"
        "fn make(x: f32) -> Pair {\n    var p: Pair;\n    p.a = x;\n    p.b = x * 2.0;\n    return p;\n}\n\n"
        "fn fill(x: f32) -> Bins {\n    var b: Bins;\n    for (var k = 0; k < 3; k++) {\n"
        "        b.v[k] = x + f32(k);\n    }\n    b.n = 2;\n    return b;\n}\n\n"
        "fn spread4(x: f32) -> vec4<f32> {\n    return vec4<f32>(x, x + 1.0, x + 2.0, x + 3.0);\n}",
    ),
    "reserved-identifier": Template(
        "let device = f32(iin[i]);\nlet kernel = iin[i] & 7;\nlet half = sample(device);\n"
        "let thread = kernel * 3 + 1;\nlet output = half + f32(thread) * 32.0;\noutp[i] = output;",
        "fn sample(input: f32) -> f32 {\n    return input * 2.0;\n}",
    ),
    "sibling-redeclaration": Template(
        "var r: f32 = 0.0;\n{\n    let t = iin[i] + 1;\n    r += f32(t);\n}\n{\n    let t = f32(iin[i]) * 0.5;\n"
        "    r += t * 32.0;\n}\nfor (var k = 0; k < 2; k++) {\n    r += f32(k) * 512.0;\n}\n"
        "for (var k = 2u; k < 4u; k++) {\n    r += f32(k) * 1024.0;\n}\nif (iin[i] > 7) {\n"
        "    let t = vec2<f32>(1.0, 2.0);\n    r += t.y * 8192.0;\n} else {\n    let t = true;\n"
        "    r += select(0.0, 32768.0, t);\n}\noutp[i] = r;"
    ),
}

RUST = {
    "octal-literal": Template(
        "let a: i32 = 0o17 + 0o0;\nlet b: u32 = 0o10 + iin[i] as u32;\nlet c: i32 = (iin[i] & 0o7) * 0o11;\n"
        "outp[i] = a as f32 + b as f32 * 32.0 + c as f32 * 1024.0;"
    ),
    "literal-suffixes": Template(
        "let a = 0.5f32 + 3. + 2e+1 + 1.5_f32;\nlet b = 7u32 + iin[i] as u32;\nlet c = iin[i] as f32 * 25e-2f32;\n"
        "outp[i] = a + b as f32 * 32.0 + c * 1024.0;"
    ),
    "unsigned-negation": Template(
        "let a = iin[i] as u32;\nlet b = a.wrapping_neg();\nlet c = (a + 1).wrapping_neg().wrapping_add(5);\n"
        "let d = (!a).wrapping_add(1);\nlet e = if d == b { 1 } else { 0 };\n"
        "outp[i] = (b >> 28) as f32 + (c >> 28) as f32 * 16.0 + e as f32 * 256.0 + (c & 15) as f32 * 512.0;"
    ),
    "call-result-access": Template(
        "let x = (iin[i] & 7) as f32;\nlet a = make(x).b;\nlet s = spread4(x).wy();\n"
        "let c = fill(x).v[(iin[i] % 3) as usize];\nlet d = make(spread4(x).z).a + fill(x).n as f32;\n"
        f"outp[i] = {_ACCESS_SUM.format(x='.x', y='.y')};",
        "#[derive(Clone, Copy, Default)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}\n\n"
        "#[derive(Clone, Copy)]\nstruct Bins {\n    v: [f32; 3],\n    n: i32,\n}\n\n"
        "fn make(x: f32) -> Pair {\n    Pair { a: x, b: x * 2.0 }\n}\n\n"
        "fn fill(x: f32) -> Bins {\n    let mut b = Bins { v: [0.0; 3], n: 0 };\n    for k in 0..3 {\n"
        "        b.v[k] = x + k as f32;\n    }\n    b.n = 2;\n    b\n}\n\n"
        "fn spread4(x: f32) -> $vec4 {\n    $vec4::new(x, x + 1.0, x + 2.0, x + 3.0)\n}",
    ),
    "reserved-identifier": Template(
        "let device = iin[i] as f32;\nlet kernel = iin[i] & 7;\nlet half = sample(device);\n"
        "let var = kernel * 3 + 1;\nlet output = half + var as f32 * 32.0;\noutp[i] = output;",
        "fn sample(input: f32) -> f32 {\n    input * 2.0\n}",
    ),
    "sibling-redeclaration": Template(
        "let mut r: f32 = 0.0;\n{\n    let t = iin[i] + 1;\n    r += t as f32;\n}\n{\n    let t = iin[i] as f32 * 0.5;\n"
        "    r += t * 32.0;\n}\nfor k in 0..2 {\n    r += k as f32 * 512.0;\n}\nfor k in 2u32..4 {\n"
        "    r += k as f32 * 1024.0;\n}\nif iin[i] > 7 {\n    let t = $vec2::new(1.0, 2.0);\n    r += t.y * 8192.0;\n"
        "} else {\n    let t = true;\n    r += if t { 32768.0 } else { 0.0 };\n}\noutp[i] = r;"
    ),
}

MOJO = {
    "octal-literal": Template(
        "var a: Int32 = 0o17 + 0o0\nvar b: UInt32 = 0o10 + UInt32(iin[i])\nvar c: Int32 = (iin[i] & 0o7) * 0o11\n"
        "outp[i] = Float32(a) + Float32(b) * 32.0 + Float32(c) * 1024.0"
    ),
    "literal-suffixes": ABSENT,
    "unsigned-negation": Template(
        "var a = UInt32(iin[i])\nvar b = -a\nvar c = -(a + 1) + 5\nvar d = ~a + 1\n"
        "var e: Float32 = 1.0 if d == b else 0.0\n"
        "outp[i] = Float32(b >> 28) + Float32(c >> 28) * 16.0 + e * 256.0 + Float32(c & 15) * 512.0"
    ),
    "call-result-access": Template(
        "var x = Float32(iin[i] & 7)\nvar a = make(x).b\nvar s = spread4(x).shuffle[3, 1]()\n"
        "var c = fill(x).v[Int(iin[i] % 3)]\nvar d = make(spread4(x)[2]).a + Float32(fill(x).n)\n"
        f"outp[i] = {_ACCESS_SUM.format(x='[0]', y='[1]')}",
        "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32\n\n\n"
        "@fieldwise_init\nstruct Bins(Copyable, Movable):\n    var v: Array[Float32, 3]\n    var n: Int32\n\n\n"
        "def make(x: Float32) -> Pair:\n    return Pair(x, x * 2.0)\n\n\n"
        "def fill(x: Float32) -> Bins:\n    var b = Bins(Array[Float32, 3](fill=0.0), 0)\n    for k in range(3):\n"
        "        b.v[k] = x + Float32(k)\n    b.n = 2\n    return b^\n\n\n"
        "def spread4(x: Float32) -> $vec4:\n    return $vec4(x, x + 1.0, x + 2.0, x + 3.0)",
    ),
    "reserved-identifier": Template(
        "var device = Float32(iin[i])\nvar kernel = iin[i] & 7\nvar half = sample(device)\n"
        "var thread = kernel * 3 + 1\nvar output = half + Float32(thread) * 32.0\noutp[i] = output",
        "def sample(input: Float32) -> Float32:\n    return input * 2.0",
    ),
    "sibling-redeclaration": Template(
        "var r: Float32 = 0.0\nif iin[i] >= 0:\n    var t = iin[i] + 1\n    r += Float32(t)\nif iin[i] >= 0:\n"
        "    var t = Float32(iin[i]) * 0.5\n    r += t * 32.0\nfor k in range(2):\n    r += Float32(k) * 512.0\n"
        "for k in range(2, 4):\n    var u = UInt32(k)\n    r += Float32(u) * 1024.0\nif iin[i] > 7:\n"
        "    var t = $vec2(1.0, 2.0)\n    r += t[1] * 8192.0\nelse:\n    var t = True\n"
        "    var w: Float32 = 32768.0 if t else 0.0\n    r += w\noutp[i] = r"
    ),
}
