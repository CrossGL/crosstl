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
    "matrix-vector-product": Template(
        "$mat2 m = $mat2(fin[i], 1.0, 2.0, 3.0);\n$vec2 v = m * $vec2(1.0, fin[i]);\noutp[i] = v.y;"
    ),
    "compute": Template("outp[i] = fin[i];"),
}


def scalar(spelling: str, source: str = "iin", prelude: str = "") -> Template:
    return Template(
        f"{spelling} v = {spelling}({source}[i]);\noutp[i] = float(v * v);",
        prelude=prelude,
    )
