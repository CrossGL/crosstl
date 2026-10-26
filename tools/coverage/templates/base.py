"""The template type and the probes that read alike in every C-like language."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    body: str = ""
    declarations: str = ""
    prelude: str = ""
    params: str = ""
    program: str = ""
    entry: str = ""


ABSENT = Template(entry="<absent>")

NESTED_STRUCTS = "struct Pair {\n    float a;\n    float b;\n};\nstruct Frame {\n    Pair p;\n    float w;\n};"

# Operator probes. The integer inputs are 0 to 15 (:func:`..execution.inputs`), so a probe subtracts 8 where it needs
# a negative operand, and divides and takes remainders of non-negative operands only: GLSL leaves the remainder of a
# negative operand undefined (GLSL 4.60, section 5.9). Compilers may fuse a multiply and an add, or factor a sum of
# products, where no precise qualifier forbids it, so these probes multiply the float input by powers of two
# only, and the matrix products work on small integers, whose products and sums are exact however they are grouped.
OPERATORS = {
    "integer-arithmetic": Template("int a = iin[i];\noutp[i] = float(a * 3 - a + 2);"),
    "float-arithmetic": Template("outp[i] = fin[i] * 2.0 - fin[i] / 4.0 + 1.0;"),
    "integer-division": Template(
        "int a = iin[i] + 7;\noutp[i] = float(a / 3 + a % 5);"
    ),
    "unsigned-division": Template(
        "uint a = uint(iin[i]) + 7u;\noutp[i] = float(a / 3u + a % 5u);"
    ),
    # An arithmetic shift keeps the sign of a negative operand, and a logical one fills a set top bit with zeros.
    "signed-shift": Template(
        "int a = iin[i] - 8;\noutp[i] = float((iin[i] << 2) + (a >> 1));"
    ),
    "unsigned-shift": Template(
        "uint a = uint(iin[i]) | 2147483648u;\noutp[i] = float((a >> 28u) + (uint(iin[i]) << 3u));"
    ),
    "bitwise-logic": Template(
        "int a = iin[i];\noutp[i] = float(((a & 6) | (a ^ 5)) + ~a);"
    ),
    "negation": Template("int a = -iin[i];\noutp[i] = -fin[i] + float(a);"),
    # Each comparison adds its own power of two, so the output says which of them held. The unsigned probe compares
    # wrapped negative values, which a signed comparison would order first.
    "signed-comparison": Template(
        "int a = iin[i] - 8;\noutp[i] = (a < 3 ? 1.0 : 0.0) + (a >= -2 ? 2.0 : 0.0) + (a == 4 ? 4.0 : 0.0)"
        " + (a != 0 ? 8.0 : 0.0);"
    ),
    "unsigned-comparison": Template(
        "uint a = uint(iin[i] - 8);\noutp[i] = (a < 4u ? 1.0 : 0.0) + (a >= 2147483648u ? 2.0 : 0.0)"
        " + (a > 5u ? 4.0 : 0.0) + (a <= 2u ? 8.0 : 0.0);"
    ),
    "float-comparison": Template(
        "float f = float(iin[i] - 8) * 0.5;\noutp[i] = (f < 0.25 ? 1.0 : 0.0) + (f >= 1.0 ? 2.0 : 0.0)"
        " + (f == 1.5 ? 4.0 : 0.0) + (f != -2.0 ? 8.0 : 0.0);"
    ),
    "boolean-logic": Template(
        "bool p = iin[i] > 7;\nbool q = (iin[i] & 1) == 1;\n"
        "outp[i] = (!p ? 1.0 : 0.0) + (p == q ? 2.0 : 0.0) + (p != q ? 4.0 : 0.0);"
    ),
    # A boolean converts to 1 or 0, and a number to true when it is not zero.
    "bool-conversion": Template(
        "bool p = iin[i] > 7;\nbool q = bool(iin[i] & 2);\noutp[i] = float(p) + 2.0 * float(int(q));"
    ),
    # A postfix operator yields the value before the change and a prefix one the value after it.
    "increment-decrement": Template(
        "int a = iin[i];\nint b = a++;\nint c = ++a;\nint d = a--;\n--a;\n"
        "outp[i] = float(a * 1000 + b * 100 + c * 10 + d);"
    ),
    "float-to-integer": Template(
        "outp[i] = float(int(fin[i])) + float(uint(fin[i] + 4.0));"
    ),
    "integer-to-float": Template(
        "int a = iin[i] - 8;\noutp[i] = float(a) * 0.5 + float(uint(iin[i]));"
    ),
    "signed-to-unsigned": Template(
        "uint a = uint(iin[i] - 8);\noutp[i] = float(a >> 28u) + float(int(a) + 8);"
    ),
    "compound-assignment": Template(
        "int a = iin[i];\na += 3;\na -= 1;\na *= 5;\na /= 2;\na %= 7;\na <<= 2;\na >>= 1;\na &= 13;\na |= 16;\n"
        "a ^= 5;\noutp[i] = float(a);"
    ),
    "vector-scalar-product": Template(
        "$vec2 v = $vec2(fin[i], 1.0) * 4.0;\n$vec2 w = 0.5 * v;\noutp[i] = w.x + w.y;"
    ),
    "matrix-product": Template(
        "float f = float(iin[i]);\n$mat2 m = $mat2(f, 1.0, 2.0, 3.0);\n$mat2 n = $mat2(1.0, f, 0.5, 2.0);\n"
        "$vec2 v = (m * n) * $vec2(1.0, f);\noutp[i] = v.x + 2.0 * v.y;"
    ),
    "vector-matrix-product": Template(
        "float f = float(iin[i]);\n$mat2 m = $mat2(f, 1.0, 2.0, 3.0);\n$vec2 v = $vec2(1.0, f) * m;\n"
        "outp[i] = v.x + 2.0 * v.y;"
    ),
}
# Probes most C-like languages, CrossGL among them, write the same way.
C_FAMILY = {
    "if-else": Template(
        "if (fin[i] > 0.0) {\n    outp[i] = 1.0;\n} else {\n    outp[i] = 2.0;\n}"
    ),
    "switch": Template(
        "switch (iin[i]) {\n    case 0:\n        outp[i] = 1.0;\n        break;\n    case 1:\n"
        "        outp[i] = 3.0;\n        break;\n    default:\n        outp[i] = 2.0;\n        break;\n}"
    ),
    "for-loop": Template(
        "float acc = 0.0;\nfor (int k = 0; k < iin[i]; k++) {\n    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "while-loop": Template(
        "float acc = 0.0;\nint k = 0;\nwhile (k < iin[i]) {\n    acc += fin[i];\n    k++;\n}\noutp[i] = acc;"
    ),
    "do-while-loop": Template(
        "float acc = 0.0;\nint k = 0;\ndo {\n    acc += fin[i];\n    k++;\n} while (k < iin[i]);\noutp[i] = acc;"
    ),
    "break": Template(
        "float acc = 0.0;\nfor (int k = 0; k < 8; k++) {\n    if (k == iin[i]) {\n        break;\n    }\n"
        "    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "continue": Template(
        "float acc = 0.0;\nfor (int k = 0; k < 8; k++) {\n    if (k == iin[i]) {\n        continue;\n    }\n"
        "    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "early-return": Template(
        "if (fin[i] < 0.0) {\n    outp[i] = 0.0;\n    return;\n}\noutp[i] = fin[i];"
    ),
    "conditional-expression": Template("outp[i] = fin[i] > 0.0 ? fin[i] : 0.0;"),
    "short-circuit-logic": Template(
        "if (iin[i] > 0 && fin[i] / float(iin[i]) > 1.0) {\n    outp[i] = 1.0;\n}"
        " else if (iin[i] < 0 || fin[i] < 0.0) {\n    outp[i] = 2.0;\n}"
    ),
    "function-call": Template(
        "outp[i] = twice(fin[i]);",
        "float twice(float x) {\n    return x * 2.0;\n}",
    ),
    "out-parameter": Template(
        "float a;\nfloat b;\nsplit(fin[i], a, b);\noutp[i] = a + b;",
        "void split(float x, out float a, out float b) {\n    a = x;\n    b = x * 2.0;\n}",
    ),
    "inout-parameter": Template(
        "float a = fin[i];\nbump(a);\noutp[i] = a;",
        "void bump(inout float a) {\n    a = a + 1.0;\n}",
    ),
    "array": Template(
        "float values[4];\nfor (int k = 0; k < 4; k++) {\n    values[k] = fin[i] * float(k);\n}\n"
        "outp[i] = values[iin[i] & 3];"
    ),
    "struct": Template(
        "Pair p;\np.a = fin[i];\np.b = 1.0;\noutp[i] = p.a + p.b;",
        "struct Pair {\n    float a;\n    float b;\n};",
    ),
    "array-of-structs": Template(
        "Pair ps[2];\nps[0].a = fin[i];\nps[1].b = 2.0;\noutp[i] = ps[0].a + ps[1].b;",
        "struct Pair {\n    float a;\n    float b;\n};",
    ),
    "swizzle": Template(
        "$vec4 v = $vec4(fin[i], 1.0, 2.0, 3.0);\noutp[i] = v.wzyx.x + v.xy.y;"
    ),
    "array-initializer": Template(
        "float values[4] = {1.0, 2.0, 4.0, 8.0};\noutp[i] = values[iin[i] & 3] + float(iin[i]);"
    ),
    "nested-struct": Template(
        "Frame s;\ns.p.a = float(iin[i]);\ns.p.b = 2.0;\ns.w = 3.0;\noutp[i] = s.p.a * s.w + s.p.b;",
        NESTED_STRUCTS,
    ),
    # Each component the probe writes scales its own power of ten in the output.
    "swizzle-assignment": Template(
        "$vec4 v = $vec4(0.0, 0.0, 0.0, 0.0);\nv.y = float(iin[i]);\nv.zx = $vec2(2.0, 3.0);\n"
        "v.wy += $vec2(1.0, 0.5);\noutp[i] = v.x * 1000.0 + v.y * 100.0 + v.z * 10.0 + v.w;"
    ),
    # m[k] is column k; the column index of c depends on the input.
    "matrix-indexing": Template(
        "float f = float(iin[i]);\n$mat2 m = $mat2(f, 1.0, 2.0, 3.0);\n$vec2 c = m[iin[i] & 1];\n"
        "outp[i] = m[1][0] * 100.0 + m[0].y * 10.0 + c.x + 0.5 * c.y;"
    ),
    "matrix-vector-product": Template(
        "$mat2 m = $mat2(fin[i], 1.0, 2.0, 3.0);\n$vec2 v = m * $vec2(1.0, fin[i]);\noutp[i] = v.y;"
    ),
    "compute": Template("outp[i] = fin[i];"),
    **OPERATORS,
}


def scalar(spelling: str, source: str = "iin", prelude: str = "") -> Template:
    return Template(
        f"{spelling} v = {spelling}({source}[i]);\noutp[i] = float(v * v);",
        prelude=prelude,
    )
