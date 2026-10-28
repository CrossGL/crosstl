"""Enumeration, default-argument, member-function, generic-function, const-parameter and boolean-bitwise probes: their
C-family form (:data:`FAMILY`, as HLSL writes them) and the languages that write them differently.

GLSL, WGSL and Mojo have no enumerations, and Slang scopes an enumeration's cases to its type. Only the C++ family
and Mojo take default arguments, and only they and Rust declare functions in a structure or over a type parameter;
OpenCL C is C99 and has none of the three. HLSL member functions and templates need HLSL 2021. Slang marks a member
function that writes its structure ``[mutating]`` and constrains a generic's parameter by an interface. CUDA vectors
have no operators, so the CUDA and HIP generic probes instantiate the template on scalars only.

WGSL, Rust and Mojo parameters cannot be written, so their const-parameter probes declare plain parameters. GLSL
applies ``&``, ``|`` and ``^`` to integers only (GLSL 4.60, section 5.9); WGSL applies ``&`` and ``|`` to booleans
but not ``^``, so its probe writes ``!=`` there.
"""

from __future__ import annotations

from .base import ABSENT, PAIR, Template
from .statements import c_text

ENUM = "enum Mode {\n    Off,\n    Low = 2,\n    High\n};"
# An odd input picks Low and an even one High; the switch and the comparison each add their own power of two.
_ENUM = (
    "Mode m = (iin[i] & 1) != 0 ? Low : High;\nfloat r = 0.0;\nswitch (m) {\ncase Off:\n    r = 1.0;\n    break;\n"
    "case Low:\n    r = 2.0;\n    break;\ndefault:\n    r = 4.0;\n    break;\n}\nif (m == High) {\n    r += 8.0;\n}\n"
    "outp[i] = r + float(int(m)) * 64.0 + float(int(Off)) * 4096.0;"
)


def scoped_enum(separator: str) -> str:
    """The enumeration probe with each case qualified by its type, as ``Mode::Low`` or ``Mode.Low``."""
    body = _ENUM
    for case in ("Off", "Low", "High"):
        body = body.replace(f" {case}", f" Mode{separator}{case}").replace(
            f"({case})", f"(Mode{separator}{case})"
        )
    return body


BLEND = (
    "float blend(float a, float b = 2.0, int k = 3) {\n    return a * b + float(k);\n}"
)
_DEFAULT = "float x = float(iin[i]);\noutp[i] = blend(x) + blend(x, 4.0) * 64.0 + blend(x, 0.5, 1) * 4096.0;"

SPAN = (
    "struct Span {\n    float lo;\n    float hi;\n    float width() {\n        return hi - lo;\n    }\n"
    "    float at(float t) {\n        return lo + t * width();\n    }\n    void widen(float d) {\n        hi += d;\n    }\n};"
)
_METHOD = (
    "Span s;\ns.lo = float(iin[i]);\ns.hi = s.lo * 2.0 + 1.0;\nfloat w = s.width();\ns.widen(3.0);\n"
    "outp[i] = w + s.at(0.5) * 64.0 + s.hi * 4096.0;"
)

TWICE = "T twice(T x) {\n    return x + x;\n}"
GENERIC = (
    "float x = float(iin[i]);\nint k = iin[i] & 7;\n$vec2 v = $vec2(x, 1.0);\n$vec2 w = twice(v);\n"
    "outp[i] = twice(x) + float(twice(k)) * 64.0 + w.y * 4096.0;"
)

WEIGH = "float weigh(const float x, const $vec2 w, const Pair p) {\n    return x * w.x + w.y + p.a * p.b;\n}"
_CONST = (
    "Pair p;\np.a = float(iin[i]);\np.b = 3.0;\n"
    "outp[i] = weigh(float(iin[i] & 7), $vec2(2.0, 1.0), p) + weigh(1.0, $vec2(p.a, 0.5), p) * 64.0;"
)

_BOOL_VALUES = "bool p = iin[i] > 3;\nbool q = (iin[i] & 1) != 0;\n"
_BOOL_SUM = "outp[i] = float(a) + float(o) * 2.0 + float(x) * 4.0 + float(m) * 8.0 + float(n) * 16.0;"
_BOOL_BITWISE = (
    _BOOL_VALUES
    + "bool a = p & q;\nbool o = p | q;\nbool x = p ^ q;\nbool m = p;\nm &= q;\nbool n = q;\nn |= !p;\n"
    + _BOOL_SUM
)

FAMILY = {
    "enum-constant": Template(_ENUM, ENUM),
    "default-argument": Template(_DEFAULT, BLEND),
    "struct-method": Template(_METHOD, SPAN),
    "generic-function": Template(GENERIC, f"template <typename T>\n{TWICE}"),
    "const-parameter": Template(_CONST, f"{PAIR}\n\n{WEIGH}"),
    "bool-bitwise": Template(_BOOL_BITWISE),
}

GLSL = {
    "enum-constant": ABSENT,
    "default-argument": ABSENT,
    "struct-method": ABSENT,
    "generic-function": ABSENT,
    "bool-bitwise": ABSENT,
}

SLANG = {
    "enum-constant": Template(scoped_enum("."), ENUM),
    "struct-method": Template(
        _METHOD, SPAN.replace("    void widen", "    [mutating]\n    void widen")
    ),
    "generic-function": Template(
        GENERIC, "T twice<T : IArithmetic>(T x) {\n    return x + x;\n}"
    ),
}


def _cuda() -> dict[str, Template]:
    """The probes as CUDA and HIP write them: casts, ``__device__`` functions and ``make_float2``."""
    unsigned = "unsigned int"
    methods = SPAN
    for method in ("float width", "float at", "void widen"):
        methods = methods.replace(f"    {method}", f"    __device__ {method}")
    scalars = GENERIC.replace(
        "$vec2 v = $vec2(x, 1.0);\n$vec2 w = twice(v);",
        "float2 w = make_float2(twice(x), twice(1.0));",
    )
    return {
        "enum-constant": Template(c_text(_ENUM, unsigned), ENUM),
        "default-argument": Template(
            c_text(_DEFAULT, unsigned), "__device__ " + c_text(BLEND, unsigned)
        ),
        "struct-method": Template(c_text(_METHOD, unsigned), c_text(methods, unsigned)),
        "generic-function": Template(
            c_text(scalars, unsigned), f"template <typename T>\n__device__ {TWICE}"
        ),
        "const-parameter": Template(
            c_text(_CONST, unsigned).replace("$vec2(", "make_float2("),
            f"{PAIR}\n\n__device__ "
            + c_text(WEIGH, unsigned).replace("$vec2", "float2"),
        ),
        "bool-bitwise": Template(c_text(_BOOL_BITWISE, unsigned)),
    }


CUDA = _cuda()

_OPENCL_PAIR = "typedef struct {\n    float a;\n    float b;\n} Pair;"

OPENCL = {
    "enum-constant": Template(
        c_text(_ENUM, "uint").replace("int(", "(int)("),
        "typedef enum {\n    Off,\n    Low = 2,\n    High\n} Mode;",
    ),
    "default-argument": ABSENT,
    "struct-method": ABSENT,
    "generic-function": ABSENT,
    "const-parameter": Template(
        c_text(_CONST, "uint").replace("$vec2(", "(float2)("),
        f"{_OPENCL_PAIR}\n\n" + c_text(WEIGH, "uint").replace("$vec2", "float2"),
    ),
    "bool-bitwise": Template(c_text(_BOOL_BITWISE, "uint")),
}

_WGSL_PAIR = "struct Pair {\n    a: f32,\n    b: f32,\n}"

WGSL = {
    "enum-constant": ABSENT,
    "default-argument": ABSENT,
    "struct-method": ABSENT,
    "generic-function": ABSENT,
    "const-parameter": Template(
        "var p: Pair;\np.a = f32(iin[i]);\np.b = 3.0;\n"
        "outp[i] = weigh(f32(iin[i] & 7), vec2<f32>(2.0, 1.0), p) + weigh(1.0, vec2<f32>(p.a, 0.5), p) * 64.0;",
        f"{_WGSL_PAIR}\n\nfn weigh(x: f32, w: vec2<f32>, p: Pair) -> f32 {{\n    return x * w.x + w.y + p.a * p.b;\n}}",
    ),
    "bool-bitwise": Template(
        "let p = iin[i] > 3;\nlet q = (iin[i] & 1) != 0;\nlet a = p & q;\nlet o = p | q;\nlet x = p != q;\n"
        "var m = p;\nm &= q;\nvar n = q;\nn |= !p;\n"
        + _BOOL_SUM.replace("float(", "f32(")
    ),
}

_RUST_PAIR = "#[derive(Clone, Copy)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}"

RUST = {
    "enum-constant": Template(
        "let m = if (iin[i] & 1) != 0 { Mode::Low } else { Mode::High };\n"
        "let mut r: f32 = match m {\n    Mode::Off => 1.0,\n    Mode::Low => 2.0,\n    _ => 4.0,\n};\n"
        "if m == Mode::High {\n    r += 8.0;\n}\n"
        "outp[i] = r + (m as i32) as f32 * 64.0 + (Mode::Off as i32) as f32 * 4096.0;",
        "#[derive(Clone, Copy, PartialEq)]\n#[repr(i32)]\nenum Mode {\n    Off,\n    Low = 2,\n    High,\n}",
    ),
    "default-argument": ABSENT,
    "struct-method": Template(
        "let mut s = Span { lo: iin[i] as f32, hi: 0.0 };\ns.hi = s.lo * 2.0 + 1.0;\nlet w = s.width();\n"
        "s.widen(3.0);\noutp[i] = w + s.at(0.5) * 64.0 + s.hi * 4096.0;",
        "#[derive(Clone, Copy)]\nstruct Span {\n    lo: f32,\n    hi: f32,\n}\n\nimpl Span {\n"
        "    fn width(&self) -> f32 {\n        self.hi - self.lo\n    }\n\n"
        "    fn at(&self, t: f32) -> f32 {\n        self.lo + t * self.width()\n    }\n\n"
        "    fn widen(&mut self, d: f32) {\n        self.hi += d;\n    }\n}",
    ),
    "generic-function": Template(
        "let x = iin[i] as f32;\nlet k = iin[i] & 7;\nlet v = $vec2::new(x, 1.0);\nlet w = twice(v);\n"
        "outp[i] = twice(x) + twice(k) as f32 * 64.0 + w.y * 4096.0;",
        "fn twice<T: Copy + core::ops::Add<Output = T>>(x: T) -> T {\n    x + x\n}",
    ),
    "const-parameter": Template(
        "let p = Pair { a: iin[i] as f32, b: 3.0 };\n"
        "outp[i] = weigh((iin[i] & 7) as f32, $vec2::new(2.0, 1.0), p) + weigh(1.0, $vec2::new(p.a, 0.5), p) * 64.0;",
        f"{_RUST_PAIR}\n\nfn weigh(x: f32, w: $vec2, p: Pair) -> f32 {{\n    x * w.x + w.y + p.a * p.b\n}}",
    ),
    "bool-bitwise": Template(
        "let p = iin[i] > 3;\nlet q = (iin[i] & 1) != 0;\nlet a = p & q;\nlet o = p | q;\nlet x = p ^ q;\n"
        "let mut m = p;\nm &= q;\nlet mut n = q;\nn |= !p;\n"
        "outp[i] = a as i32 as f32 + (o as i32 as f32) * 2.0 + (x as i32 as f32) * 4.0 + (m as i32 as f32) * 8.0\n"
        "    + (n as i32 as f32) * 16.0;"
    ),
}

_MOJO_PAIR = "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32"

MOJO = {
    "enum-constant": ABSENT,
    "default-argument": Template(
        "var x = Float32(iin[i])\noutp[i] = blend(x) + blend(x, 4.0) * 64.0 + blend(x, 0.5, 1) * 4096.0",
        "def blend(a: Float32, b: Float32 = 2.0, k: Int32 = 3) -> Float32:\n    return a * b + Float32(k)",
    ),
    "struct-method": Template(
        "var s = Span(Float32(iin[i]), 0.0)\ns.hi = s.lo * 2.0 + 1.0\nvar w = s.width()\ns.widen(3.0)\n"
        "outp[i] = w + s.at(0.5) * 64.0 + s.hi * 4096.0",
        "@fieldwise_init\nstruct Span(Copyable, Movable):\n    var lo: Float32\n    var hi: Float32\n\n"
        "    def width(self) -> Float32:\n        return self.hi - self.lo\n\n"
        "    def at(self, t: Float32) -> Float32:\n        return self.lo + t * self.width()\n\n"
        "    def widen(mut self, d: Float32):\n        self.hi += d",
    ),
    "generic-function": Template(
        "var x = Float32(iin[i])\nvar k = iin[i] & 7\nvar v = $vec2(x, 1.0)\nvar w = twice(v)\n"
        "outp[i] = twice(x) + Float32(twice(k)) * 64.0 + w[1] * 4096.0",
        "def twice(x: SIMD) -> type_of(x):\n    return x + x",
    ),
    "const-parameter": Template(
        "var p = Pair(Float32(iin[i]), 3.0)\n"
        "outp[i] = weigh(Float32(iin[i] & 7), $vec2(2.0, 1.0), p) + weigh(1.0, $vec2(p.a, 0.5), p) * 64.0",
        f"{_MOJO_PAIR}\n\n\ndef weigh(x: Float32, w: $vec2, p: Pair) -> Float32:\n    return x * w[0] + w[1] + p.a * p.b",
    ),
    "bool-bitwise": Template(
        "var p = iin[i] > 3\nvar q = (iin[i] & 1) != 0\nvar a = p & q\nvar o = p | q\nvar x = p ^ q\nvar m = p\n"
        "m &= q\nvar n = q\nn |= not p\n"
        "outp[i] = Float32(Int(a)) + Float32(Int(o)) * 2.0 + Float32(Int(x)) * 4.0 + Float32(Int(m)) * 8.0"
        " + Float32(Int(n)) * 16.0"
    ),
}
