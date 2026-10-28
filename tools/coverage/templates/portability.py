"""Nested-unary, condition-assignment, constant-expression, intrinsic-name, control-attribute and array-return
probes: their C-family form (:data:`FAMILY`, as HLSL writes them) and the languages that write them differently.

The nested-unary probe derives its float from an integer, so every product and sum in it is exact: the comparison with
the native probe is bit for bit, and a driver that reassociates float arithmetic would otherwise round the native probe
and a translation that spells a negation differently to different results.

WGSL and Rust have no unary ``+`` and Rust complements an integer with ``!``. Only the C family and GLSL assign or
increment inside a condition: WGSL and Mojo assign in a statement before the test, and Rust in a block that ends in
it. Rust and CUDA build a constant vector component by component.

Each intrinsic-name probe names its four helpers with words that are built-in functions in other languages but not in
its own, and gives them other arities and results, so a translation that calls the built-in computes something else.

HLSL and Slang mark loops ``[unroll]`` or ``[loop]`` and branches ``[branch]`` or ``[flatten]``; GLSL writes the same
four hints with ``GL_EXT_control_flow_attributes``. Metal, CUDA, HIP and OpenCL C hint only loops, with ``#pragma
unroll``, and Mojo unrolls one with ``comptime for``. GLSL ES 3.00, WGSL and Rust have no such hints.

HLSL returns an array through a ``typedef``, Metal returns ``array<T, N>`` and Mojo ``Array``; C and C++ functions
cannot return an array at all (C99 6.7.5.3; C++17 [dcl.fct] paragraph 11).
"""

from __future__ import annotations

from .base import ABSENT, Template
from .statements import c_text

_UNARY = (
    "int k = iin[i];\nfloat x = float(k) * 0.5 - 3.75;\nbool b = k > 3;\nfloat a = -(-x);\nfloat c = x - -x * 0.5;\n"
    "float d = +x + -(+x) * 0.25;\nint e = ~(~k) + -(~k);\nint f = k - -k;\nbool g = !(!b);\n"
    "outp[i] = a + c * 4.0 + d * 16.0 + float(e) * 64.0 + float(f) * 2048.0 + (g ? 65536.0 : 0.0);"
)

# Each condition changes a variable before its comparison reads it: k and m by an increment, j by a prefix one and t
# by a compound assignment, once per evaluation of the condition.
_CONDITION = (
    "int lim = iin[i] & 7;\nint k = 0;\nint n = 0;\nwhile (k++ < lim) {\n    n += 2;\n}\nint j = iin[i];\n"
    "float r = 0.0;\nif (++j > 8) {\n    r = 1.0;\n}\nint m = 3;\nfor (; m-- > 0;) {\n    r += 2.0;\n}\n"
    "int t = 20;\nwhile ((t -= 3 + (lim & 1)) > 0) {\n    n += 64;\n}\n"
)
_CONDITION_SUM = (
    "outp[i] = float(n) + float(k) * 512.0 + float(j) * 8192.0 + r * 262144.0 + float(m) * 2097152.0"
    " + float(t) * 4194304.0;"
)


def constants(qualifier: str) -> str:
    """The program-scope constants of the constant-expression probe, each computed from the one before it."""
    return (
        f"{qualifier} int N = 3;\n{qualifier} int M = N * 2 + 1;\n{qualifier} float S = float(M) * 0.5;\n"
        f"{qualifier} $vec2 V = $vec2(1.0, 2.0) * S;"
    )


_CONSTANT = (
    "$vec2 w = V * fin[i];\noutp[i] = fin[i] * S + w.y * 16.0 + float(M + N) * 256.0;"
)


def intrinsic_names(names: tuple[str, str, str, str]) -> Template:
    """The intrinsic-name probe in C-family syntax with its four helpers called ``names``."""
    one, two, three, four = names
    return Template(
        f"float x = fin[i];\noutp[i] = {one}(x) + {two}(x, 1.0) * 16.0 + {three}(x) * 256.0 + {four}(x) * 4096.0;",
        f"float {one}(float x) {{\n    return x * 0.5 + 1.0;\n}}\n\n"
        f"float {two}(float a, float b) {{\n    return a * 2.0 - b;\n}}\n\n"
        f"float {three}(float x) {{\n    return x + 0.25;\n}}\n\n"
        f"float {four}(float x) {{\n    return x * x;\n}}",
    )


# Words that are built-in functions in CUDA, Metal or HLSL but not in GLSL or CrossGL, which spells them abs, mix, fract
# and inversesqrt (tools/coverage/spellings.py). CrossGL has saturate, so its probe cannot declare that one.
SHADER_NAMES = ("fabs", "lerp", "frac", "rsqrt")


def attributed(unroll: str, loop: str, branch: str, flatten: str) -> str:
    """The control-attribute probe with each hint written as the line before its statement; an empty hint is left out."""
    unroll, loop, branch, flatten = (
        f"{h}\n" if h else "" for h in (unroll, loop, branch, flatten)
    )
    return (
        f"float acc = 0.0;\n{unroll}for (int k = 0; k < 4; k++) {{\n    acc += float(k);\n}}\n"
        f"{loop}for (int j = 0; j < (iin[i] & 7); j++) {{\n    acc += 8.0;\n}}\n"
        f"{branch}if (iin[i] > 7) {{\n    acc += 128.0;\n}}\n"
        f"{flatten}if ((iin[i] & 1) == 0) {{\n    acc += 256.0;\n}} else {{\n    acc -= 512.0;\n}}\n"
        "outp[i] = acc + fin[i] * 2048.0;"
    )


_ARRAY_SUM = "outp[i] = a[0] + a[1] * 16.0 + a[2] * 256.0 + spread(2.0)[k] * 4096.0;"
TRIPLE = "typedef float Triple[3];\n\nTriple spread(float x) {\n    Triple r = {x, x * 2.0, x + 1.0};\n    return r;\n}"
GLSL_SPREAD = "float[3] spread(float x) {\n    return float[3](x, x * 2.0, x + 1.0);\n}"
GLSL_ARRAY = f"float a[3] = spread(fin[i]);\nint k = iin[i] % 3;\n{_ARRAY_SUM}"

FAMILY = {
    "nested-unary": Template(_UNARY),
    "condition-assignment": Template(_CONDITION + _CONDITION_SUM),
    "constant-expression": Template(_CONSTANT, constants("static const")),
    "intrinsic-name": intrinsic_names(("fract", "mix", "mod", "inversesqrt")),
    "control-attribute": Template(
        attributed("[unroll]", "[loop]", "[branch]", "[flatten]")
    ),
    "array-return": Template(
        f"Triple a = spread(fin[i]);\nint k = iin[i] % 3;\n{_ARRAY_SUM}", TRIPLE
    ),
}

GLSL = {
    "constant-expression": Template(_CONSTANT, constants("const")),
    "intrinsic-name": intrinsic_names(SHADER_NAMES),
    "control-attribute": Template(
        attributed("[[unroll]]", "[[dont_unroll]]", "[[dont_flatten]]", "[[flatten]]"),
        prelude="#extension GL_EXT_control_flow_attributes : require",
    ),
    "array-return": Template(GLSL_ARRAY, GLSL_SPREAD),
}

WEBGL = {"control-attribute": ABSENT}

METAL = {
    "constant-expression": Template(_CONSTANT, constants("constant")),
    "intrinsic-name": intrinsic_names(("frac", "lerp", "mod", "inversesqrt")),
    "control-attribute": Template(
        attributed("#pragma unroll", "#pragma nounroll", "", "")
    ),
    "array-return": Template(
        f"array<float, 3> a = spread(fin[i]);\nint k = iin[i] % 3;\n{_ARRAY_SUM}",
        "array<float, 3> spread(float x) {\n    return {x, x * 2.0, x + 1.0};\n}",
    ),
}


def _c(unsigned: str, device: str) -> dict[str, Template]:
    """The probes in C, with casts, ``unsigned`` for uint and ``device`` before each helper function."""
    names = intrinsic_names(("saturate", "lerp", "frac", "mix"))
    helpers = c_text(names.declarations, unsigned).replace(
        "\nfloat ", f"\n{device}float "
    )
    return {
        "nested-unary": Template(c_text(_UNARY, unsigned)),
        "condition-assignment": Template(c_text(_CONDITION + _CONDITION_SUM, unsigned)),
        "intrinsic-name": Template(c_text(names.body, unsigned), device + helpers),
        "control-attribute": Template(
            c_text(attributed("#pragma unroll", "#pragma unroll 1", "", ""), unsigned)
        ),
        "array-return": ABSENT,
    }


CUDA = {
    **_c("unsigned int", "__device__ "),
    "constant-expression": Template(
        c_text(_CONSTANT, "unsigned int").replace(
            "$vec2 w = V * fin[i];",
            "float2 w = make_float2(V.x * fin[i], V.y * fin[i]);",
        ),
        "constexpr int N = 3;\nconstexpr int M = N * 2 + 1;\nconstexpr float S = (float)(M) * 0.5f;\n"
        "__constant__ float2 V = {1.0f * S, 2.0f * S};",
    ),
}

OPENCL = {
    **_c("uint", ""),
    "intrinsic-name": Template(
        c_text(
            intrinsic_names(("saturate", "lerp", "frac", "inversesqrt")).body, "uint"
        ),
        c_text(
            intrinsic_names(("saturate", "lerp", "frac", "inversesqrt")).declarations,
            "uint",
        ),
    ),
    "constant-expression": Template(
        c_text(_CONSTANT, "uint").replace("$vec2 w", "float2 w"),
        c_text(constants("__constant"), "uint")
        .replace("$vec2(", "(float2)(")
        .replace("$vec2", "float2"),
    ),
}

WGSL = {
    "nested-unary": Template(
        "let k = iin[i];\nlet x = f32(k) * 0.5 - 3.75;\nlet b = k > 3;\nlet a = -(-x);\nlet c = x - -x * 0.5;\n"
        "let d = x + -(x) * 0.25;\nlet e = ~(~k) + -(~k);\nlet f = k - -k;\nlet g = !(!b);\n"
        "outp[i] = a + c * 4.0 + d * 16.0 + f32(e) * 64.0 + f32(f) * 2048.0 + select(0.0, 65536.0, g);"
    ),
    "condition-assignment": Template(
        "let lim = iin[i] & 7;\nvar k = 0;\nvar n = 0;\nloop {\n    let c = k;\n    k++;\n    if !(c < lim) {\n"
        "        break;\n    }\n    n += 2;\n}\nvar j = iin[i];\nvar r = 0.0;\nj++;\nif j > 8 {\n    r = 1.0;\n}\n"
        "var m = 3;\nloop {\n    let c = m;\n    m--;\n    if !(c > 0) {\n        break;\n    }\n    r += 2.0;\n}\n"
        "var t = 20;\nloop {\n    t -= 3 + (lim & 1);\n    if !(t > 0) {\n        break;\n    }\n    n += 64;\n}\n"
        + _CONDITION_SUM.replace("float(", "f32(")
    ),
    "constant-expression": Template(
        _CONSTANT.replace("float(", "f32(").replace("$vec2 w", "let w"),
        "const N: i32 = 3;\nconst M: i32 = N * 2 + 1;\nconst S: f32 = f32(M) * 0.5;\n"
        "const V: vec2<f32> = vec2<f32>(1.0, 2.0) * S;",
    ),
    "intrinsic-name": Template(
        "let x = fin[i];\noutp[i] = frac(x) + lerp(x, 1.0) * 16.0 + rsqrt(x) * 256.0 + inversesqrt(x) * 4096.0;",
        "fn frac(x: f32) -> f32 {\n    return x * 0.5 + 1.0;\n}\n\n"
        "fn lerp(a: f32, b: f32) -> f32 {\n    return a * 2.0 - b;\n}\n\n"
        "fn rsqrt(x: f32) -> f32 {\n    return x + 0.25;\n}\n\n"
        "fn inversesqrt(x: f32) -> f32 {\n    return x * x;\n}",
    ),
    "control-attribute": ABSENT,
    "array-return": Template(
        f"let a = spread(fin[i]);\nlet k = iin[i] % 3;\n{_ARRAY_SUM}",
        "fn spread(x: f32) -> array<f32, 3> {\n    return array<f32, 3>(x, x * 2.0, x + 1.0);\n}",
    ),
}

RUST = {
    "nested-unary": Template(
        "let k = iin[i];\nlet x = k as f32 * 0.5 - 3.75;\nlet b = k > 3;\nlet a = -(-x);\nlet c = x - -x * 0.5;\n"
        "let d = x + -(x) * 0.25;\nlet e = !(!k) + -(!k);\nlet f = k - -k;\nlet g = !(!b);\n"
        "outp[i] = a + c * 4.0 + d * 16.0 + e as f32 * 64.0 + f as f32 * 2048.0 + if g { 65536.0 } else { 0.0 };"
    ),
    "condition-assignment": Template(
        "let lim = iin[i] & 7;\nlet mut k = 0;\nlet mut n = 0;\nwhile {\n    k += 1;\n    k - 1 < lim\n} {\n"
        "    n += 2;\n}\nlet mut j = iin[i];\nlet mut r: f32 = 0.0;\nif {\n    j += 1;\n    j > 8\n} {\n"
        "    r = 1.0;\n}\nlet mut m = 3;\nwhile {\n    m -= 1;\n    m + 1 > 0\n} {\n    r += 2.0;\n}\n"
        "let mut t = 20;\nwhile {\n    t -= 3 + (lim & 1);\n    t > 0\n} {\n    n += 64;\n}\n"
        "outp[i] = n as f32 + k as f32 * 512.0 + j as f32 * 8192.0 + r * 262144.0 + m as f32 * 2097152.0\n"
        "    + t as f32 * 4194304.0;"
    ),
    "constant-expression": Template(
        "let w = V * fin[i];\noutp[i] = fin[i] * S + w.y * 16.0 + (M + N) as f32 * 256.0;",
        "const N: i32 = 3;\nconst M: i32 = N * 2 + 1;\nconst S: f32 = M as f32 * 0.5;\n"
        "const V: $vec2 = $vec2::new(1.0 * S, 2.0 * S);",
    ),
    "intrinsic-name": Template(
        "let x = fin[i];\noutp[i] = saturate(x) + lerp(x, 1.0) * 16.0 + frac(x) * 256.0 + rsqrt(x) * 4096.0;",
        "fn saturate(x: f32) -> f32 {\n    x * 0.5 + 1.0\n}\n\nfn lerp(a: f32, b: f32) -> f32 {\n    a * 2.0 - b\n}\n\n"
        "fn frac(x: f32) -> f32 {\n    x + 0.25\n}\n\nfn rsqrt(x: f32) -> f32 {\n    x * x\n}",
    ),
    "control-attribute": ABSENT,
    "array-return": Template(
        f"let a = spread(fin[i]);\nlet k = (iin[i] % 3) as usize;\n{_ARRAY_SUM}",
        "fn spread(x: f32) -> [f32; 3] {\n    [x, x * 2.0, x + 1.0]\n}",
    ),
}

MOJO = {
    "nested-unary": Template(
        "var k = iin[i]\nvar x = Float32(k) * 0.5 - 3.75\nvar b = k > 3\nvar a = -(-x)\nvar c = x - -x * 0.5\n"
        "var d = +x + -(+x) * 0.25\nvar e = ~(~k) + -(~k)\nvar f = k - -k\nvar g = not (not b)\n"
        "outp[i] = a + c * 4.0 + d * 16.0 + Float32(e) * 64.0 + Float32(f) * 2048.0 + Float32(Int(g)) * 65536.0"
    ),
    "condition-assignment": Template(
        "var lim = iin[i] & 7\nvar k: Int32 = 0\nvar n: Int32 = 0\nwhile True:\n    var c = k\n    k += 1\n"
        "    if not (c < lim):\n        break\n    n += 2\nvar j = iin[i]\nvar r: Float32 = 0.0\nj += 1\nif j > 8:\n"
        "    r = 1.0\nvar m: Int32 = 3\nwhile True:\n    var d = m\n    m -= 1\n    if not (d > 0):\n        break\n"
        "    r += 2.0\nvar t: Int32 = 20\nwhile True:\n    t -= 3 + (lim & 1)\n    if not (t > 0):\n        break\n"
        "    n += 64\noutp[i] = Float32(n) + Float32(k) * 512.0 + Float32(j) * 8192.0 + r * 262144.0"
        " + Float32(m) * 2097152.0 + Float32(t) * 4194304.0"
    ),
    "constant-expression": Template(
        "var w = V * fin[i]\noutp[i] = fin[i] * S + w[1] * 16.0 + Float32(M + N) * 256.0",
        "comptime N: Int32 = 3\ncomptime M: Int32 = N * 2 + 1\ncomptime S: Float32 = Float32(M) * 0.5\n"
        "comptime V = $vec2(1.0, 2.0) * S",
    ),
    "intrinsic-name": Template(
        "var x = fin[i]\noutp[i] = saturate(x) + lerp(x, 1.0) * 16.0 + frac(x) * 256.0 + rsqrt(x) * 4096.0",
        "def saturate(x: Float32) -> Float32:\n    return x * 0.5 + 1.0\n\n\n"
        "def lerp(a: Float32, b: Float32) -> Float32:\n    return a * 2.0 - b\n\n\n"
        "def frac(x: Float32) -> Float32:\n    return x + 0.25\n\n\n"
        "def rsqrt(x: Float32) -> Float32:\n    return x * x",
    ),
    "control-attribute": Template(
        "var acc: Float32 = 0.0\ncomptime for k in range(4):\n    acc += Float32(k)\n"
        "for _ in range(Int(iin[i] & 7)):\n    acc += 8.0\nif iin[i] > 7:\n    acc += 128.0\n"
        "if (iin[i] & 1) == 0:\n    acc += 256.0\nelse:\n    acc -= 512.0\noutp[i] = acc + fin[i] * 2048.0"
    ),
    "array-return": Template(
        f"var a = spread(fin[i])\nvar k = Int(iin[i] % 3)\n{_ARRAY_SUM.removesuffix(';')}",
        "def spread(x: Float32) -> Array[Float32, 3]:\n    return [x, x * 2.0, x + 1.0]",
    ),
}
