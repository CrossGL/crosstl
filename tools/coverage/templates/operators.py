"""Operator probes in the languages that do not write them as the C family does (:data:`.base.OPERATORS`).

C converts with casts, HLSL multiplies matrices with ``mul`` and fills them by rows, WGSL shifts by unsigned amounts,
Mojo divides integers with ``//`` and has no matrices, and Rust converts with ``as`` and complements with ``!``.
"""

from __future__ import annotations

from .base import ABSENT, OPERATORS, Template


def c_casts(unsigned: str) -> dict[str, Template]:
    """The operator probes in C, which converts with casts rather than constructors; ``unsigned`` is the
    language's spelling of uint. Float literals take the ``f`` suffix, so float arithmetic stays single precision.
    """
    return {
        "integer-arithmetic": Template(
            "int a = iin[i];\noutp[i] = (float)(a * 3 - a + 2);"
        ),
        "float-arithmetic": Template("outp[i] = fin[i] * 2.0f - fin[i] / 4.0f + 1.0f;"),
        "integer-division": Template(
            "int a = iin[i] + 7;\noutp[i] = (float)(a / 3 + a % 5);"
        ),
        "unsigned-division": Template(
            f"{unsigned} a = ({unsigned})iin[i] + 7u;\noutp[i] = (float)(a / 3u + a % 5u);"
        ),
        "signed-shift": Template(
            "int a = iin[i] - 8;\noutp[i] = (float)((iin[i] << 2) + (a >> 1));"
        ),
        "unsigned-shift": Template(
            f"{unsigned} a = ({unsigned})iin[i] | 2147483648u;\n"
            f"outp[i] = (float)((a >> 28u) + (({unsigned})iin[i] << 3u));"
        ),
        "bitwise-logic": Template(
            "int a = iin[i];\noutp[i] = (float)(((a & 6) | (a ^ 5)) + ~a);"
        ),
        "negation": Template("int a = -iin[i];\noutp[i] = -fin[i] + (float)a;"),
        "float-to-integer": Template(
            f"outp[i] = (float)(int)fin[i] + (float)({unsigned})(fin[i] + 4.0f);"
        ),
        "integer-to-float": Template(
            f"int a = iin[i] - 8;\noutp[i] = (float)a * 0.5f + (float)({unsigned})iin[i];"
        ),
        "signed-to-unsigned": Template(
            f"{unsigned} a = ({unsigned})(iin[i] - 8);\noutp[i] = (float)(a >> 28u) + (float)((int)a + 8);"
        ),
        "compound-assignment": Template(
            OPERATORS["compound-assignment"].body.replace("float(a)", "(float)a")
        ),
    }


# The scalars are listed by rows, as HLSL fills a matrix, to make the CrossGL probe's matrices.
HLSL = {
    "matrix-product": Template(
        "float f = float(iin[i]);\nfloat2x2 m = float2x2(f, 2.0, 1.0, 3.0);\nfloat2x2 n = float2x2(1.0, 0.5, f, 2.0);\n"
        "float2 v = mul(mul(m, n), float2(1.0, f));\noutp[i] = v.x + 2.0 * v.y;"
    ),
    "vector-matrix-product": Template(
        "float f = float(iin[i]);\nfloat2x2 m = float2x2(f, 2.0, 1.0, 3.0);\nfloat2 v = mul(float2(1.0, f), m);\n"
        "outp[i] = v.x + 2.0 * v.y;"
    ),
}

WGSL = {
    "integer-arithmetic": Template("let a = iin[i];\noutp[i] = f32(a * 3 - a + 2);"),
    "float-arithmetic": OPERATORS["float-arithmetic"],
    "integer-division": Template("let a = iin[i] + 7;\noutp[i] = f32(a / 3 + a % 5);"),
    "unsigned-division": Template(
        "let a = u32(iin[i]) + 7u;\noutp[i] = f32(a / 3u + a % 5u);"
    ),
    # WGSL shifts by an unsigned amount only.
    "signed-shift": Template(
        "let a = iin[i] - 8;\noutp[i] = f32((iin[i] << 2u) + (a >> 1u));"
    ),
    "unsigned-shift": Template(
        "let a = u32(iin[i]) | 2147483648u;\noutp[i] = f32((a >> 28u) + (u32(iin[i]) << 3u));"
    ),
    "bitwise-logic": Template(
        "let a = iin[i];\noutp[i] = f32(((a & 6) | (a ^ 5)) + ~a);"
    ),
    "negation": Template("let a = -iin[i];\noutp[i] = -fin[i] + f32(a);"),
    "float-to-integer": Template(
        "outp[i] = f32(i32(fin[i])) + f32(u32(fin[i] + 4.0));"
    ),
    "integer-to-float": Template(
        "let a = iin[i] - 8;\noutp[i] = f32(a) * 0.5 + f32(u32(iin[i]));"
    ),
    "signed-to-unsigned": Template(
        "let a = u32(iin[i] - 8);\noutp[i] = f32(a >> 28u) + f32(i32(a) + 8);"
    ),
    "compound-assignment": Template(
        "var a = iin[i];\na += 3;\na -= 1;\na *= 5;\na /= 2;\na %= 7;\na <<= 2u;\na >>= 1u;\na &= 13;\n"
        "a |= 16;\na ^= 5;\noutp[i] = f32(a);"
    ),
    "vector-scalar-product": Template(
        "let v = vec2<f32>(fin[i], 1.0) * 4.0;\nlet w = 0.5 * v;\noutp[i] = w.x + w.y;"
    ),
    "matrix-product": Template(
        "let f = f32(iin[i]);\nlet m = mat2x2<f32>(f, 1.0, 2.0, 3.0);\nlet n = mat2x2<f32>(1.0, f, 0.5, 2.0);\n"
        "let v = (m * n) * vec2<f32>(1.0, f);\noutp[i] = v.x + 2.0 * v.y;"
    ),
    "vector-matrix-product": Template(
        "let f = f32(iin[i]);\nlet m = mat2x2<f32>(f, 1.0, 2.0, 3.0);\nlet v = vec2<f32>(1.0, f) * m;\n"
        "outp[i] = v.x + 2.0 * v.y;"
    ),
}

MOJO = {
    "integer-arithmetic": Template("var a = iin[i]\noutp[i] = Float32(a * 3 - a + 2)"),
    "float-arithmetic": Template("outp[i] = fin[i] * 2.0 - fin[i] / 4.0 + 1.0"),
    # // floors; on the non-negative operands the probes divide, flooring and truncating agree.
    "integer-division": Template(
        "var a = iin[i] + 7\noutp[i] = Float32(a // 3 + a % 5)"
    ),
    "unsigned-division": Template(
        "var a = UInt32(iin[i]) + 7\noutp[i] = Float32(a // 3 + a % 5)"
    ),
    "signed-shift": Template(
        "var a = iin[i] - 8\noutp[i] = Float32((iin[i] << 2) + (a >> 1))"
    ),
    "unsigned-shift": Template(
        "var a = UInt32(iin[i]) | 2147483648\noutp[i] = Float32((a >> 28) + (UInt32(iin[i]) << 3))"
    ),
    "bitwise-logic": Template(
        "var a = iin[i]\noutp[i] = Float32(((a & 6) | (a ^ 5)) + ~a)"
    ),
    "negation": Template("var a = -iin[i]\noutp[i] = -fin[i] + Float32(a)"),
    "float-to-integer": Template(
        "outp[i] = Float32(Int32(fin[i])) + Float32(UInt32(fin[i] + 4.0))"
    ),
    "integer-to-float": Template(
        "var a = iin[i] - 8\noutp[i] = Float32(a) * 0.5 + Float32(UInt32(iin[i]))"
    ),
    "signed-to-unsigned": Template(
        "var a = UInt32(iin[i] - 8)\noutp[i] = Float32(a >> 28) + Float32(Int32(a) + 8)"
    ),
    "compound-assignment": Template(
        "var a = iin[i]\na += 3\na -= 1\na *= 5\na //= 2\na %= 7\na <<= 2\na >>= 1\na &= 13\na |= 16\n"
        "a ^= 5\noutp[i] = Float32(a)"
    ),
    "vector-scalar-product": Template(
        "var v = $vec2(fin[i], 1.0) * 4.0\nvar w = 0.5 * v\noutp[i] = w[0] + w[1]"
    ),
    "matrix-product": ABSENT,
    "vector-matrix-product": ABSENT,
}

RUST = {
    "integer-arithmetic": Template(
        "let a = iin[i];\noutp[i] = (a * 3 - a + 2) as f32;"
    ),
    "float-arithmetic": OPERATORS["float-arithmetic"],
    "integer-division": Template(
        "let a = iin[i] + 7;\noutp[i] = (a / 3 + a % 5) as f32;"
    ),
    "unsigned-division": Template(
        "let a = iin[i] as u32 + 7;\noutp[i] = (a / 3 + a % 5) as f32;"
    ),
    "signed-shift": Template(
        "let a = iin[i] - 8;\noutp[i] = ((iin[i] << 2) + (a >> 1)) as f32;"
    ),
    "unsigned-shift": Template(
        "let a = iin[i] as u32 | 2147483648;\noutp[i] = ((a >> 28) + ((iin[i] as u32) << 3)) as f32;"
    ),
    "bitwise-logic": Template(
        "let a = iin[i];\noutp[i] = (((a & 6) | (a ^ 5)) + !a) as f32;"
    ),
    "negation": Template("let a = -iin[i];\noutp[i] = -fin[i] + a as f32;"),
    "float-to-integer": Template(
        "outp[i] = fin[i] as i32 as f32 + (fin[i] + 4.0) as u32 as f32;"
    ),
    "integer-to-float": Template(
        "let a = iin[i] - 8;\noutp[i] = a as f32 * 0.5 + iin[i] as u32 as f32;"
    ),
    "signed-to-unsigned": Template(
        "let a = (iin[i] - 8) as u32;\noutp[i] = (a >> 28) as f32 + (a as i32 + 8) as f32;"
    ),
    "compound-assignment": Template(
        "let mut a = iin[i];\na += 3;\na -= 1;\na *= 5;\na /= 2;\na %= 7;\na <<= 2;\na >>= 1;\na &= 13;\n"
        "a |= 16;\na ^= 5;\noutp[i] = a as f32;"
    ),
    "vector-scalar-product": Template(
        "let v = $vec2::new(fin[i], 1.0) * 4.0;\nlet w = 0.5 * v;\noutp[i] = w.x + w.y;"
    ),
    "matrix-product": Template(
        "let f = iin[i] as f32;\nlet m = $mat2::from_cols($vec2::new(f, 1.0), $vec2::new(2.0, 3.0));\n"
        "let n = $mat2::from_cols($vec2::new(1.0, f), $vec2::new(0.5, 2.0));\n"
        "let v = (m * n) * $vec2::new(1.0, f);\noutp[i] = v.x + 2.0 * v.y;"
    ),
    # glam multiplies a matrix by a column vector only; it has no row-vector product.
    "vector-matrix-product": ABSENT,
}
