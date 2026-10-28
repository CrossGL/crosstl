"""Comma-operator, aggregate-equality, integer-condition, global-shadowing, local-struct and type-alias probes: their
C-family form (:data:`FAMILY`, as HLSL writes them) and the languages that write them differently.

Only the C family has a comma operator; WGSL, Rust and Mojo have none. CrossGL's ``==`` and ``!=`` compare vectors
componentwise and structures and arrays as a whole (ADR 0002). GLSL compares all three as a whole (GLSL 4.60, section
5.9), so its probe writes ``a == b`` where the others write ``all(a == b)``. The languages with no equality on
structures or arrays compare their members, and so does the Rust probe on arrays, whose equality rust-gpu cannot
compile because it goes through slices.

An integer stands for a condition only in the C family and Mojo; GLSL, WGSL and Rust take only ``bool`` conditions.
Rust cannot shadow a constant with a binding (E0005), so its global-shadowing probe is absent. WGSL and Mojo declare
structures only at module scope, and GLSL has no ``typedef``.
"""

from __future__ import annotations

from .base import ABSENT, PAIR, Template
from .statements import c_text

_COMMA = (
    "int a = iin[i];\nint b = (a += 3, a * 2);\nint c = 0;\nfloat r = b > 20 ? (c = 1, 2.0) : (c = 3, 4.0);\n"
    "int d = (c++, c + a);\noutp[i] = float(b) + r * 64.0 + float(c) * 512.0 + float(d) * 4096.0;"
)

# Six tests, two each on the vectors, the structures and the arrays; each that holds adds its own power of two.
_EQUALITY_TESTS = (
    "float r = 0.0;\nif ({0}) {{\n    r += 1.0;\n}}\nif ({1}) {{\n    r += 2.0;\n}}\nif ({2}) {{\n    r += 4.0;\n}}\n"
    "if ({3}) {{\n    r += 8.0;\n}}\nif ({4}) {{\n    r += 16.0;\n}}\nif ({5}) {{\n    r += 32.0;\n}}\noutp[i] = r;"
)
_EQUALITY_VALUES = (
    "$vec2 a = $vec2(float(iin[i] & 3), 1.0);\n$vec2 b = $vec2(2.0, 1.0);\nPair p;\np.a = float(iin[i]);\np.b = 2.0;\n"
    "Pair q;\nq.a = 5.0;\nq.b = p.b;\nfloat t[2];\nt[0] = float(iin[i] & 1);\nt[1] = 3.0;\nfloat u[2];\n"
    "u[0] = 1.0;\nu[1] = 3.0;\n"
)
_MEMBERWISE = (
    "p.a == q.a && p.b == q.b",
    "p.a != q.a || p.b != q.b",
    "t[0] == u[0] && t[1] == u[1]",
    "t[0] != u[0] || t[1] != u[1]",
)
_EQUALITY = _EQUALITY_VALUES + _EQUALITY_TESTS.format(
    "all(a == b)", "any(a != b)", *_MEMBERWISE
)
# CrossGL compares structures and arrays as a whole, GLSL vectors too.
CROSSGL_EQUALITY = _EQUALITY_VALUES + _EQUALITY_TESTS.format(
    "all(a == b)", "any(a != b)", "p == q", "p != q", "t == u", "t != u"
)
_GLSL_EQUALITY = _EQUALITY_VALUES + _EQUALITY_TESTS.format(
    "a == b", "a != b", "p == q", "p != q", "t == u", "t != u"
)

_INT_CONDITION = (
    "int n = iin[i] & 3;\nint m = iin[i] >> 2;\nuint u = uint(iin[i]) & 4u;\nfloat r = 0.0;\nif (n) {\n    r += 1.0;\n}\n"
    "if (!m) {\n    r += 2.0;\n}\nif (n && m) {\n    r += 4.0;\n}\nif (u || !n) {\n    r += 8.0;\n}\n"
    "r += (m & 1) ? 16.0 : 0.0;\nint k = n;\nwhile (k) {\n    r += 32.0;\n    k--;\n}\noutp[i] = r;"
)
# The same conditions written as comparisons, for CrossGL.
CROSSGL_CONDITION = (
    _INT_CONDITION.replace("if (n)", "if (n != 0)")
    .replace("if (!m)", "if (m == 0)")
    .replace("if (n && m)", "if (n != 0 && m != 0)")
    .replace("if (u || !n)", "if (u != 0u || n == 0)")
    .replace("(m & 1) ?", "(m & 1) != 0 ?")
    .replace("while (k)", "while (k != 0)")
)

# apply's parameter and local reuse the names of the constant and of twice, and so do a block's local and the body's.
SHADOW_HELPERS = (
    "float twice(float x) {\n    return x * scale;\n}\n\nfloat apply(float scale) {\n    float twice = scale + 1.0;\n"
    "    return twice * 4.0;\n}"
)
_SHADOW = (
    "float r = twice(float(iin[i])) + apply(float(iin[i] & 3));\n{\n    float scale = 8.0;\n    r += scale * 64.0;\n}\n"
    "float twice = r;\noutp[i] = twice + scale * 4096.0;"
)

_LOCAL_STRUCT = (
    "struct Span {\n    float lo;\n    float hi;\n};\nSpan s;\ns.lo = float(iin[i]);\ns.hi = s.lo + 4.0;\nSpan t = s;\n"
    "t.hi *= 2.0;\noutp[i] = s.hi + t.hi * 64.0 + t.lo * 4096.0;"
)

ALIASES = (
    "typedef float real;\ntypedef $vec2 duo;\ntypedef int count;\ntypedef Pair Couple;"
)
ALIAS_HELPER = "real scaled(duo w, count k) {\n    return w.x * w.y + real(k);\n}"
_ALIAS_SUM = "scaled(v, n) + c.a * 64.0 + c.b * 4096.0"
ALIAS = (
    "real x = float(iin[i]);\nduo v = duo(x, 2.0);\ncount n = iin[i] & 3;\nCouple c;\nc.a = x;\nc.b = 1.0;\n"
    f"outp[i] = {_ALIAS_SUM};"
)

FAMILY = {
    "comma-operator": Template(_COMMA),
    "aggregate-equality": Template(_EQUALITY, PAIR),
    "integer-condition": Template(_INT_CONDITION),
    "global-shadowing": Template(
        _SHADOW, f"static const float scale = 2.0;\n\n{SHADOW_HELPERS}"
    ),
    "local-struct": Template(_LOCAL_STRUCT),
    "type-alias": Template(ALIAS, f"{PAIR}\n{ALIASES}\n\n{ALIAS_HELPER}"),
}

GLSL = {
    "aggregate-equality": Template(_GLSL_EQUALITY, PAIR),
    "integer-condition": ABSENT,
    "global-shadowing": Template(
        _SHADOW, f"const float scale = 2.0;\n\n{SHADOW_HELPERS}"
    ),
    "type-alias": ABSENT,
}

METAL = {
    "global-shadowing": Template(
        _SHADOW, f"constant float scale = 2.0;\n\n{SHADOW_HELPERS}"
    ),
}


def _c_family(
    unsigned: str, device: str, constant: str, vector: str, alias: str
) -> dict[str, Template]:
    """The C family's probes as CUDA and HIP (``device`` ``__device__ ``) and OpenCL C (``device`` empty) write them.

    ``constant`` declares the shadowed constant, ``vector`` builds a ``float2`` and ``alias`` a ``duo``:
    ``make_float2(`` in CUDA, whose vectors have no operators, and ``(float2)(`` and ``(duo)(`` in OpenCL C, whose
    structures need a ``typedef`` to go unprefixed.
    """
    if device:
        pair = PAIR
        span = "struct Span {\n    float lo;\n    float hi;\n};"
        vectors = ("a.x == b.x && a.y == b.y", "a.x != b.x || a.y != b.y")
    else:
        pair = "typedef struct {\n    float a;\n    float b;\n} Pair;"
        span = "typedef struct {\n    float lo;\n    float hi;\n} Span;"
        vectors = ("all(a == b)", "any(a != b)")
    equality = _EQUALITY_VALUES + _EQUALITY_TESTS.format(*vectors, *_MEMBERWISE)
    helper = ALIAS_HELPER if device else ALIAS_HELPER.replace("real(k)", "(real)k")
    return {
        "comma-operator": Template(c_text(_COMMA, unsigned)),
        "aggregate-equality": Template(
            c_text(equality, unsigned)
            .replace("$vec2(", vector)
            .replace("$vec2", "float2"),
            pair,
        ),
        "integer-condition": Template(c_text(_INT_CONDITION, unsigned)),
        "global-shadowing": Template(
            c_text(_SHADOW, unsigned),
            f"{constant}\n\n{device}"
            + c_text(SHADOW_HELPERS, unsigned).replace(
                "\n\nfloat apply", f"\n\n{device}float apply"
            ),
        ),
        "local-struct": Template(
            c_text(_LOCAL_STRUCT, unsigned).replace(
                "struct Span {\n    float lo;\n    float hi;\n};", span
            )
        ),
        "type-alias": Template(
            c_text(ALIAS, unsigned).replace("duo(", alias),
            f"{pair}\n"
            + ALIASES.replace("$vec2", "float2")
            + f"\n\n{device}"
            + c_text(helper, unsigned),
        ),
    }


CUDA = _c_family(
    "unsigned int",
    "__device__ ",
    "constexpr float scale = 2.0f;",
    "make_float2(",
    "make_float2(",
)
OPENCL = _c_family("uint", "", "__constant float scale = 2.0f;", "(float2)(", "(duo)(")

_WGSL_PAIR = "struct Pair {\n    a: f32,\n    b: f32,\n}"

WGSL = {
    "comma-operator": ABSENT,
    "aggregate-equality": Template(
        "let a = vec2<f32>(f32(iin[i] & 3), 1.0);\nlet b = vec2<f32>(2.0, 1.0);\nvar p: Pair;\np.a = f32(iin[i]);\n"
        "p.b = 2.0;\nvar q: Pair;\nq.a = 5.0;\nq.b = p.b;\nvar t: array<f32, 2>;\nt[0] = f32(iin[i] & 1);\n"
        "t[1] = 3.0;\nvar u = array<f32, 2>(1.0, 3.0);\nvar r: f32 = 0.0;\n"
        + _EQUALITY_TESTS.format("all(a == b)", "any(a != b)", *_MEMBERWISE).split(
            "\n", 1
        )[1],
        _WGSL_PAIR,
    ),
    "integer-condition": ABSENT,
    "global-shadowing": Template(
        "var r = twice(f32(iin[i])) + apply(f32(iin[i] & 3));\n{\n    let scale = 8.0;\n    r += scale * 64.0;\n}\n"
        "let twice = r;\noutp[i] = twice + scale * 4096.0;",
        "const scale: f32 = 2.0;\n\nfn twice(x: f32) -> f32 {\n    return x * scale;\n}\n\n"
        "fn apply(scale: f32) -> f32 {\n    let twice = scale + 1.0;\n    return twice * 4.0;\n}",
    ),
    "local-struct": ABSENT,
    "type-alias": Template(
        "let x: real = real(iin[i]);\nlet v = duo(x, 2.0);\nlet n: count = iin[i] & 3;\nvar c: Couple;\nc.a = x;\n"
        f"c.b = 1.0;\noutp[i] = {_ALIAS_SUM};",
        f"{_WGSL_PAIR}\n\nalias real = f32;\nalias duo = vec2<f32>;\nalias count = i32;\nalias Couple = Pair;\n\n"
        "fn scaled(w: duo, k: count) -> real {\n    return w.x * w.y + real(k);\n}",
    ),
}

_RUST_PAIR = (
    "#[derive(Clone, Copy, PartialEq)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}"
)

RUST = {
    "comma-operator": ABSENT,
    "aggregate-equality": Template(
        "let a = $vec2::new((iin[i] & 3) as f32, 1.0);\nlet b = $vec2::new(2.0, 1.0);\n"
        "let p = Pair { a: iin[i] as f32, b: 2.0 };\nlet q = Pair { a: 5.0, b: p.b };\n"
        "let t = [(iin[i] & 1) as f32, 3.0];\nlet u = [1.0f32, 3.0];\nlet mut r = 0.0;\n"
        "if a == b {\n    r += 1.0;\n}\nif a != b {\n    r += 2.0;\n}\nif p == q {\n    r += 4.0;\n}\n"
        "if p != q {\n    r += 8.0;\n}\nif t[0] == u[0] && t[1] == u[1] {\n    r += 16.0;\n}\n"
        "if t[0] != u[0] || t[1] != u[1] {\n    r += 32.0;\n}\noutp[i] = r;",
        _RUST_PAIR,
    ),
    "integer-condition": ABSENT,
    "global-shadowing": ABSENT,
    "local-struct": Template(
        "#[derive(Clone, Copy)]\nstruct Span {\n    lo: f32,\n    hi: f32,\n}\n"
        "let mut s = Span { lo: iin[i] as f32, hi: 0.0 };\ns.hi = s.lo + 4.0;\nlet mut t = s;\nt.hi *= 2.0;\n"
        "outp[i] = s.hi + t.hi * 64.0 + t.lo * 4096.0;"
    ),
    "type-alias": Template(
        "let x: Real = iin[i] as Real;\nlet v = Duo::new(x, 2.0);\nlet n: Count = iin[i] & 3;\n"
        f"let c = Couple {{ a: x, b: 1.0 }};\noutp[i] = {_ALIAS_SUM};",
        f"{_RUST_PAIR}\n\ntype Real = f32;\ntype Duo = $vec2;\ntype Count = i32;\ntype Couple = Pair;\n\n"
        "fn scaled(w: Duo, k: Count) -> Real {\n    w.x * w.y + k as Real\n}",
    ),
}

_MOJO_PAIR = "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32"

MOJO = {
    "comma-operator": ABSENT,
    "aggregate-equality": Template(
        "var a = $vec2(Float32(iin[i] & 3), 1.0)\nvar b = $vec2(2.0, 1.0)\nvar p = Pair(Float32(iin[i]), 2.0)\n"
        "var q = Pair(5.0, p.b)\nvar t = Array[Float32, 2](fill=3.0)\n"
        "t[0] = Float32(iin[i] & 1)\nvar u = Array[Float32, 2](fill=3.0)\nu[0] = 1.0\nvar r: Float32 = 0.0\nif a.eq(b).reduce_and():\n    r += 1.0\n"
        "if a.ne(b).reduce_or():\n    r += 2.0\nif p.a == q.a and p.b == q.b:\n    r += 4.0\n"
        "if p.a != q.a or p.b != q.b:\n    r += 8.0\nif t[0] == u[0] and t[1] == u[1]:\n    r += 16.0\n"
        "if t[0] != u[0] or t[1] != u[1]:\n    r += 32.0\noutp[i] = r",
        _MOJO_PAIR,
    ),
    "integer-condition": Template(
        "var n = iin[i] & 3\nvar m = iin[i] >> 2\nvar u = UInt32(iin[i]) & 4\nvar r: Float32 = 0.0\nif n:\n    r += 1.0\n"
        "if not m:\n    r += 2.0\nif n and m:\n    r += 4.0\nif u or not n:\n    r += 8.0\n"
        "var w: Float32 = 16.0 if m & 1 else 0.0\nr += w\nvar k = n\nwhile k:\n    r += 32.0\n    k -= 1\noutp[i] = r"
    ),
    "global-shadowing": Template(
        "var r = twice(Float32(iin[i])) + apply(Float32(iin[i] & 3))\nif iin[i] >= 0:\n    var scale: Float32 = 8.0\n"
        "    r += scale * 64.0\nvar twice = r\noutp[i] = twice + scale * 4096.0",
        "comptime scale: Float32 = 2.0\n\n\ndef twice(x: Float32) -> Float32:\n    return x * scale\n\n\n"
        "def apply(scale: Float32) -> Float32:\n    var twice = scale + 1.0\n    return twice * 4.0",
    ),
    "local-struct": ABSENT,
    "type-alias": Template(
        "var x: Real = Real(iin[i])\nvar v = Duo(x, 2.0)\nvar n: Count = iin[i] & 3\nvar c = Couple(x, 1.0)\n"
        "outp[i] = scaled(v, n) + c.a * 64.0 + c.b * 4096.0",
        f"{_MOJO_PAIR}\n\n\ncomptime Real = Float32\ncomptime Duo = $vec2\ncomptime Count = Int32\ncomptime Couple = Pair\n\n\n"
        "def scaled(w: Duo, k: Count) -> Real:\n    return w[0] * w[1] + Real(k)",
    ),
}
