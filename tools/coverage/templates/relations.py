"""Integer-vector-comparison, bool-vector-logic and bool-vector-conversion probes: relational operators on signed and
unsigned integer vectors, negation and comparison of boolean vectors, and conversions between boolean vectors and
numeric ones.

HLSL, Metal, Slang, WGSL, OpenCL C and CrossGL compare vectors with the C operators, and the comparison yields a
boolean vector; GLSL spells each comparison as a function, ``lessThan`` to ``notEqual``, and negates with ``not``.
glam compares with ``cmplt`` and its relatives, has no vector equality that yields a vector, so its probe compares
boolean vectors with ``^``, and converts with ``From``, except to a float vector: rust-gpu lowers ``f32::from(bool)``
through an 8-bit integer, which needs the Int8 capability, so that conversion is a ``select``. Mojo compares SIMD
values with ``lt`` and its relatives and converts with ``cast``. OpenCL C has no boolean vectors: its comparisons
yield ``int`` vectors of -1 and 0. CUDA and HIP vector types have no operators.

Each probe adds a power of two for each test that holds. The unsigned comparisons convert a negative input to a
large unsigned value, so a translation that compares the operands as signed integers sets other bits.
"""

from __future__ import annotations

from .base import ABSENT, Template
from .statements import c_text

NAMES = ("integer-vector-comparison", "bool-vector-logic", "bool-vector-conversion")

_SETUP = "$ivec2 a = $ivec2(iin[i], -3);\n$ivec2 b = $ivec2(4, iin[i]);\n$uvec2 u = $uvec2(uint(iin[i]), 7u);\n"
# The comparisons as C writes them; GLSL's functions take the same operands.
_RELATIONS = (
    ("lt", "<", "lessThan", "a", "b"),
    ("ge", ">=", "greaterThanEqual", "a", "b"),
    ("gt", ">", "greaterThan", "u", "$uvec2(5u, 7u)"),
    ("le", "<=", "lessThanEqual", "u", "$uvec2(9u, 6u)"),
    ("eq", "==", "equal", "a", "$ivec2(iin[i], -3)"),
    ("ne", "!=", "notEqual", "u", "$uvec2(3u, 7u)"),
)
_COMPARISON_TESTS = (
    "lt.x",
    "lt.y",
    "any(ge)",
    "gt.x",
    "any(le)",
    "all(eq)",
    "ne.x",
    "all(ne)",
)
_LOGIC_SETUP = "$bvec2 b = $bvec2(fin[i] < 0.0, iin[i] > 2);\n$bvec2 c = $bvec2((iin[i] & 1) == 0, true);\n"
_LOGIC_TESTS = ("n.x", "n.y", "e.x", "all(e)", "any(d)", "d.y", "any(!c)")
_CONVERSION_SUM = (
    "float(j.x + j.y * 2) + g.x * 4.0 + g.y * 8.0 + float(u.x * 16u + u.y * 32u)"
)


def _score(
    tests: tuple[str, ...], line: str = "if ({}) {{\n    r += {}.0;\n}}\n"
) -> str:
    """Add 2^k to ``r`` when the k-th test holds, then store ``r``."""
    body = "".join(line.format(test, 1 << k) for k, test in enumerate(tests))
    return f"float r = 0.0;\n{body}outp[i] = r;"


def _comparison(glsl: bool) -> str:
    lines = (
        (
            f"$bvec2 {name} = {function}({left}, {right});\n"
            if glsl
            else f"$bvec2 {name} = {left} {operator} {right};\n"
        )
        for name, operator, function, left, right in _RELATIONS
    )
    return _SETUP + "".join(lines) + _score(_COMPARISON_TESTS)


_C_LOGIC = (
    _LOGIC_SETUP
    + "$bvec2 n = !b;\n$bvec2 e = b == c;\n$bvec2 d = b != c;\n"
    + _score(_LOGIC_TESTS)
)
_GLSL_LOGIC = (
    _LOGIC_SETUP
    + "$bvec2 n = not(b);\n$bvec2 e = equal(b, c);\n$bvec2 d = notEqual(b, c);\n"
    + _score(_LOGIC_TESTS).replace("any(!c)", "any(not(c))")
)
_CONVERSION = (
    "$ivec2 k = $ivec2(iin[i], 0);\n$vec2 f = $vec2(fin[i], 0.0);\n$bvec2 b = $bvec2(k);\n$bvec2 c = $bvec2(f);\n"
    "$ivec2 j = $ivec2(c);\n$vec2 g = $vec2(b);\n$uvec2 u = $uvec2($bvec2(fin[i] > 1.0, true));\n"
    f"outp[i] = {_CONVERSION_SUM};"
)

# GLSL; WebGL 2 has the same functions and constructors.
GLSL = {
    "integer-vector-comparison": Template(_comparison(glsl=True)),
    "bool-vector-logic": Template(_GLSL_LOGIC),
    "bool-vector-conversion": Template(_CONVERSION),
}
CROSSGL = {
    "integer-vector-comparison": Template(_comparison(glsl=False)),
    "bool-vector-logic": Template(_C_LOGIC),
    "bool-vector-conversion": Template(_CONVERSION),
}
# HLSL converts a vector to another element type with a cast.
HLSL = {
    "integer-vector-comparison": Template(_comparison(glsl=False)),
    "bool-vector-logic": Template(_C_LOGIC),
    "bool-vector-conversion": Template(
        "int2 k = int2(iin[i], 0);\nfloat2 f = float2(fin[i], 0.0);\nbool2 b = (bool2)k;\nbool2 c = (bool2)f;\n"
        "int2 j = (int2)c;\nfloat2 g = (float2)b;\nuint2 u = (uint2)bool2(fin[i] > 1.0, true);\n"
        f"outp[i] = {_CONVERSION_SUM};"
    ),
}
METAL = CROSSGL

WGSL = {
    "integer-vector-comparison": Template(
        "let a = vec2<i32>(iin[i], -3);\nlet b = vec2<i32>(4, iin[i]);\nlet u = vec2<u32>(u32(iin[i]), 7u);\n"
        + "".join(
            f"let {name} = {left} {operator} {right};\n"
            for name, operator, _, left, right in _RELATIONS
        )
        .replace("$uvec2(", "vec2<u32>(")
        .replace("$ivec2(", "vec2<i32>(")
        + _score(_COMPARISON_TESTS).replace("float r = 0.0;", "var r = 0.0;")
    ),
    "bool-vector-logic": Template(
        "let b = vec2<bool>(fin[i] < 0.0, iin[i] > 2);\nlet c = vec2<bool>((iin[i] & 1) == 0, true);\n"
        "let n = !b;\nlet e = b == c;\nlet d = b != c;\n"
        + _score(_LOGIC_TESTS).replace("float r = 0.0;", "var r = 0.0;")
    ),
    "bool-vector-conversion": Template(
        "let k = vec2<i32>(iin[i], 0);\nlet f = vec2<f32>(fin[i], 0.0);\nlet b = vec2<bool>(k);\n"
        "let c = vec2<bool>(f);\nlet j = vec2<i32>(c);\nlet g = vec2<f32>(b);\n"
        "let u = vec2<u32>(vec2<bool>(fin[i] > 1.0, true));\n"
        f"outp[i] = {_CONVERSION_SUM.replace('float(', 'f32(')};"
    ),
}

# OpenCL C compares into int vectors, whose components are -1 when the comparison holds.
OPENCL = {
    "integer-vector-comparison": Template(
        c_text(
            "int2 a = (int2)(iin[i], -3);\nint2 b = (int2)(4, iin[i]);\nuint2 u = (uint2)(uint(iin[i]), 7u);\n"
            + "".join(
                f"int2 {name} = {left} {operator} {right};\n"
                for name, operator, _, left, right in _RELATIONS
            )
            .replace("$uvec2(", "(uint2)(")
            .replace("$ivec2(", "(int2)(")
            + _score(_COMPARISON_TESTS),
            "uint",
        )
    ),
    "bool-vector-logic": ABSENT,
    "bool-vector-conversion": ABSENT,
}

CUDA = dict.fromkeys(NAMES, ABSENT)

_R = "if {} {{\n    r += {}.0;\n}}\n"
_RUST_TESTS = {
    "any(ge)": "ge.any()",
    "any(le)": "le.any()",
    "all(eq)": "eq.all()",
    "all(ne)": "ne.all()",
}
_RUST_LOGIC = {"all(e)": "e.all()", "any(d)": "d.any()", "any(!c)": "(!c).any()"}


def _rust_score(tests: tuple[str, ...], names: dict[str, str]) -> str:
    return _score(tuple(names.get(t, t) for t in tests), _R).replace(
        "float r = 0.0;", "let mut r = 0.0;"
    )


RUST = {
    "integer-vector-comparison": Template(
        "let a = IVec2::new(iin[i], -3);\nlet b = IVec2::new(4, iin[i]);\nlet u = UVec2::new(iin[i] as u32, 7);\n"
        "let lt = a.cmplt(b);\nlet ge = a.cmpge(b);\nlet gt = u.cmpgt(UVec2::new(5, 7));\n"
        "let le = u.cmple(UVec2::new(9, 6));\nlet eq = a.cmpeq(IVec2::new(iin[i], -3));\n"
        "let ne = u.cmpne(UVec2::new(3, 7));\n"
        + _rust_score(_COMPARISON_TESTS, _RUST_TESTS)
    ),
    # glam's == compares whole vectors, so the component-wise equality is the complement of ^.
    "bool-vector-logic": Template(
        "let b = BVec2::new(fin[i] < 0.0, iin[i] > 2);\nlet c = BVec2::new((iin[i] & 1) == 0, true);\n"
        "let n = !b;\nlet e = !(b ^ c);\nlet d = b ^ c;\n"
        + _rust_score(_LOGIC_TESTS, _RUST_LOGIC)
    ),
    "bool-vector-conversion": Template(
        "let k = IVec2::new(iin[i], 0);\nlet f = Vec2::new(fin[i], 0.0);\nlet b = k.cmpne(IVec2::ZERO);\n"
        "let c = f.cmpne(Vec2::ZERO);\nlet j = IVec2::from(c);\nlet g = Vec2::select(b, Vec2::ONE, Vec2::ZERO);\n"
        "let u = UVec2::from(BVec2::new(fin[i] > 1.0, true));\n"
        "outp[i] = (j.x + j.y * 2) as f32 + g.x * 4.0 + g.y * 8.0 + (u.x * 16 + u.y * 32) as f32;"
    ),
}

_M = "if {}:\n    r += {}.0\n"
_MOJO_NAMES = {
    "lt.x": "lt[0]",
    "lt.y": "lt[1]",
    "gt.x": "gt[0]",
    "ne.x": "ne[0]",
    "any(ge)": "ge.reduce_or()",
    "any(le)": "le.reduce_or()",
    "all(eq)": "eq.reduce_and()",
    "all(ne)": "ne.reduce_and()",
    "n.x": "n[0]",
    "n.y": "n[1]",
    "e.x": "e[0]",
    "d.y": "d[1]",
    "all(e)": "e.reduce_and()",
    "any(d)": "d.reduce_or()",
    "any(!c)": "(~c).reduce_or()",
}


def _mojo_score(tests: tuple[str, ...]) -> str:
    body = "".join(_M.format(_MOJO_NAMES[t], 1 << k) for k, t in enumerate(tests))
    return f"var r: Float32 = 0.0\n{body}outp[i] = r"


_MOJO_BOOL = "SIMD[DType.bool, 2]"
MOJO = {
    "integer-vector-comparison": Template(
        "var a = $ivec2(iin[i], -3)\nvar b = $ivec2(4, iin[i])\nvar u = $uvec2(UInt32(iin[i]), 7)\n"
        "var lt = a.lt(b)\nvar ge = a.ge(b)\nvar gt = u.gt($uvec2(5, 7))\nvar le = u.le($uvec2(9, 6))\n"
        "var eq = a.eq($ivec2(iin[i], -3))\nvar ne = u.ne($uvec2(3, 7))\n"
        + _mojo_score(_COMPARISON_TESTS)
    ),
    "bool-vector-logic": Template(
        f"var b = {_MOJO_BOOL}(fin[i] < 0.0, iin[i] > 2)\nvar c = {_MOJO_BOOL}((iin[i] & 1) == 0, True)\n"
        "var n = ~b\nvar e = b.eq(c)\nvar d = b.ne(c)\n" + _mojo_score(_LOGIC_TESTS)
    ),
    "bool-vector-conversion": Template(
        "var k = $ivec2(iin[i], 0)\nvar f = $vec2(fin[i], 0.0)\nvar b = k.cast[DType.bool]()\n"
        "var c = f.cast[DType.bool]()\nvar j = c.cast[DType.int32]()\nvar g = b.cast[DType.float32]()\n"
        f"var u = {_MOJO_BOOL}(fin[i] > 1.0, True).cast[DType.uint32]()\n"
        "outp[i] = Float32(j[0] + j[1] * 2) + g[0] * 4.0 + g[1] * 8.0 + Float32(u[0] * 16 + u[1] * 32)"
    ),
}
