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

PAIR = "struct Pair {\n    float a;\n    float b;\n};"
NESTED_STRUCTS = PAIR + "\nstruct Frame {\n    Pair p;\n    float w;\n};"
MAKE_PAIR = "Pair make(float x) {\n    Pair p;\n    p.a = x;\n    p.b = x * 2.0;\n    return p;\n}"
BINS = "struct Bins {\n    float v[3];\n    int n;\n};"
FIND = (
    "float find(int n, float x) {\n    for (int k = 0; k < 8; k++) {\n        if (k == n) {\n"
    "            return x + float(k);\n        }\n    }\n    return -1.0;\n}"
)
TWICE = "float twice(float x) {\n    return x * 2.0;\n}"
BLEND = "float blend(float x, float y) {\n    return x + y * 0.5;\n}"
SPREAD = "float spread(Pair q) {\n    q.a = q.a * 2.0;\n    return q.a + q.b;\n}"
POSE = "struct Pose {\n    $vec4 v;\n    $vec2 u;\n    int n;\n};"
# quadruple calls twice above its definition, which the prototype declares.
QUAD = "float twice(float x);\n\nfloat quadruple(float x) {\n    return twice(twice(x));\n}\n\n"
# bump and leap change their argument, so a conditional expression that calls both changes it twice.
STEPS = (
    "float bump(inout int k) {\n    k = k + 1;\n    return float(k);\n}\n\n"
    "float leap(inout int k) {\n    k = k + 16;\n    return float(k) * 2.0;\n}"
)
# Three paths return from inside the switch; case 2, and the default when a is 8 or less, return after it.
PICK = (
    "float pick(int a) {\n    switch (a & 3) {\n        case 0:\n            return 1.0;\n        case 1:\n"
    "            return float(a) * 2.0;\n        case 2:\n            break;\n        default:\n"
    "            if (a > 8) {\n                return 64.0;\n            }\n            break;\n    }\n"
    "    return 4.0 + float(a);\n}"
)
# advance and slot change their argument, so each call gives a new value.
ADVANCE = "int advance(inout int k) {\n    k = k + 5;\n    return k;\n}"
SLOT = "int slot(inout int k) {\n    k = k + 1;\n    return k;\n}"
BAG = (
    "struct Item {\n    $vec2 pos;\n    float w[2];\n};\n"
    "struct Bag {\n    Item items[2];\n    int n;\n};"
)
# The sum of the members a nested-member-assignment probe writes, weighted so that each one stays recoverable.
BAG_SUM = (
    "b.items[0].pos.y + b.items[1].pos.y * 32.0 + b.items[0].w[1] * 1024.0 + b.items[1].w[0] * 16384.0"
    " + b.items[0].w[0] * 262144.0 + b.items[1].w[1] * 4194304.0"
)
# Fills the named float[4] with iin[i], iin[i] + 1, iin[i] + 2 and iin[i] + 3.
_FILL = "for (int k = 0; k < 4; k++) {{\n    {0}[k] = float(iin[i] + k);\n}}\n"

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
    # Unsigned arithmetic wraps modulo 2^32: b comes back to the input, and c and d keep their wrapped top bits.
    "unsigned-wraparound": Template(
        "uint a = 4294967295u;\nuint n = uint(iin[i]);\nuint b = a + n + 1u;\nuint c = n * 2654435761u;\n"
        "uint d = n - 16u;\noutp[i] = float(b) + float(c >> 28u) * 16.0 + float(d >> 28u) * 256.0;"
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
    # An array argument is copied, so the callee's write to v[0] must not reach values[0].
    "array-parameter": Template(
        "float values[4];\n"
        + _FILL.format("values")
        + "outp[i] = total(values) * 10.0 + values[0];",
        "float total(float v[4]) {\n    v[0] = 0.0;\n    return v[0] + v[1] + v[2] + v[3];\n}",
    ),
    "struct-return": Template(
        "Pair p = make(fin[i]);\noutp[i] = p.a + p.b * 4.0;", PAIR + "\n" + MAKE_PAIR
    ),
    # The int overload scales by 4 and the float one by 2, so picking the wrong one changes the result.
    "function-overloading": Template(
        "outp[i] = scaled(fin[i]) + scaled(iin[i]);",
        "float scaled(float x) {\n    return x * 2.0;\n}\nfloat scaled(int x) {\n    return float(x) * 4.0;\n}",
    ),
    # Case 0 falls through into case 1, and cases 2 and 3 share their statements.
    "switch-fallthrough": Template(
        "float acc = 0.0;\nswitch (iin[i] & 3) {\n    case 0:\n        acc += 1.0;\n    case 1:\n"
        "        acc += 2.0;\n        break;\n    case 2:\n    case 3:\n        acc += 4.0;\n        break;\n}\n"
        "outp[i] = acc;"
    ),
    # Each inner x hides the outer one only inside its own block, so the last line reads the input again.
    "block-scope": Template(
        "float x = fin[i];\nfloat r = 0.0;\n{\n    float x = 2.0;\n    r += x;\n}\n"
        "for (int k = 0; k < 2; k++) {\n    float x = float(k + 1) * 10.0;\n    r += x;\n}\n"
        "outp[i] = r * 100.0 + x;"
    ),
    # A private global keeps one value per invocation; bump() reads and writes the kernel's copy.
    "global-variable": Template(
        "float before = counter;\ncounter = fin[i];\nbump();\noutp[i] = counter * 10.0 + before;",
        "float counter = 1.0;\n\nvoid bump() {\n    counter = counter * 2.0 + 1.0;\n}",
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
    # Assigning an array copies it, so the write to b[0] must not reach a[0].
    "array-copy": Template(
        "float a[4];\n"
        + _FILL.format("a")
        + "float b[4] = a;\nb[0] = 9.0;\noutp[i] = a[0] * 10.0 + b[0] + b[3];"
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
    # v takes a vector and two scalars, s splats one scalar, and each component of v scales its own power of ten.
    "vector-constructor": Template(
        "float f = fin[i];\n$vec2 h = $vec2(f, 1.0);\n$vec4 v = $vec4(h, 2.0, f);\n$vec3 s = $vec3(f + 1.0);\n"
        "outp[i] = v.x * 1000.0 + v.y * 100.0 + v.z * 10.0 + v.w + s.z * 0.5;"
    ),
    # The columns are (f, 1) and (2, 3), so m[0][1] is 1 and m[1][0] is 2.
    "matrix-from-vectors": Template(
        "float f = float(iin[i]);\n$mat2 m = $mat2($vec2(f, 1.0), $vec2(2.0, 3.0));\n"
        "outp[i] = m[0][1] * 100.0 + m[1][0] * 10.0 + m[0][0];"
    ),
    # lt is (fin[i] < 0, iin[i] < 8) and eq is (true, iin[i] == 4); each test that holds adds its own power of two.
    "vector-comparison": Template(
        "$vec2 a = $vec2(fin[i], float(iin[i]));\n$bvec2 lt = a < $vec2(0.0, 8.0);\n"
        "$bvec2 eq = a == $vec2(fin[i], 4.0);\nfloat r = 0.0;\nif (lt.x) {\n    r += 1.0;\n}\n"
        "if (any(lt)) {\n    r += 2.0;\n}\nif (all(lt)) {\n    r += 4.0;\n}\nif (all(eq)) {\n    r += 8.0;\n}\n"
        "outp[i] = r;"
    ),
    "struct-constructor": Template(
        "Pair p = Pair(fin[i], 2.0);\nPair q = Pair(p.b, p.a * 3.0);\noutp[i] = p.a + q.a * 10.0 + q.b * 100.0;",
        PAIR,
    ),
    # Assigning a struct copies it, so the write to b.a must not reach a.a.
    "struct-copy": Template(
        "Pair a;\na.a = fin[i];\na.b = 2.0;\nPair b = a;\nb.a = 9.0;\noutp[i] = a.a * 10.0 + b.a + b.b;",
        PAIR,
    ),
    # a[r][c] starts as 3r + c; the input picks the row whose last element it overwrites.
    "array-of-arrays": Template(
        "float a[2][3];\nfor (int r = 0; r < 2; r++) {\n    for (int c = 0; c < 3; c++) {\n"
        "        a[r][c] = float(r * 3 + c);\n    }\n}\na[iin[i] & 1][2] = fin[i];\n"
        "outp[i] = a[1][2] * 100.0 + a[0][2] * 10.0 + a[1][0];"
    ),
    # The input picks the component the probe writes and reads.
    "vector-dynamic-index": Template(
        "$vec4 v = $vec4(1.0, 2.0, 4.0, 8.0);\nint k = iin[i] & 3;\nv[k] = fin[i];\n"
        "outp[i] = v[k] * 100.0 + v[(k + 1) & 3] * 10.0 + v.x;"
    ),
    "integer-vector-arithmetic": Template(
        "$ivec2 a = $ivec2(iin[i], 3);\n$ivec2 b = a * 2 + $ivec2(1, 5);\n"
        "$ivec2 c = (b >> 1) ^ (b & $ivec2(6, 6));\n$ivec2 d = b / 3 + b % 4;\n"
        "outp[i] = float(c.x * 100 + c.y * 10 + d.x + d.y * 1000);"
    ),
    # A scalar condition picks a whole vector.
    "vector-conditional": Template(
        "$vec2 a = $vec2(fin[i], 1.0);\n$vec2 b = $vec2(2.0, fin[i] * 4.0);\n$vec2 r = iin[i] > 7 ? a : b;\n"
        "outp[i] = r.x * 16.0 + r.y;"
    ),
    # find() returns from inside its loop when it reaches n, and after the loop only for n of 8 or more.
    "loop-return": Template("outp[i] = find(iin[i], fin[i]);", FIND),
    "numeric-literals": Template(
        "int h = 0x1F + 0X0a;\nuint m = 0xFFu & uint(iin[i] * 37);\nfloat e = 2.5e-1 + 1E2 + .5;\n"
        "outp[i] = float(h) + float(m) * 64.0 + e * 4.0;"
    ),
    # The input picks the element it overwrites; the other two keep the values the loop gave them.
    "struct-array-member": Template(
        "Bins b;\nfor (int k = 0; k < 3; k++) {\n    b.v[k] = float(k + 1);\n}\nb.n = iin[i];\n"
        "b.v[iin[i] % 3] = fin[i];\noutp[i] = b.v[0] * 64.0 + b.v[1] * 8.0 + b.v[2] + float(b.n) * 512.0;",
        BINS,
    ),
    # One declaration names several variables, one of them without an initializer.
    "multiple-declarators": Template(
        "float a = fin[i], b = 2.0, c;\nc = a * 4.0 + b;\nint j = 1, k = iin[i];\n"
        "outp[i] = c + float(j + k) * 64.0;"
    ),
    # Calls as arguments of calls; blend() weights its two arguments differently, so swapping them shows.
    "nested-call": Template(
        "outp[i] = twice(blend(fin[i], 3.0)) + blend(twice(float(iin[i])), twice(fin[i]));",
        TWICE + "\n" + BLEND,
    ),
    # A struct argument is copied, so the callee's write to q.a must not reach p.a.
    "struct-parameter": Template(
        "Pair p;\np.a = fin[i];\np.b = float(iin[i]);\noutp[i] = spread(p) * 10.0 + p.a;",
        PAIR + "\n" + SPREAD,
    ),
    "const-local": Template(
        "const float k = 3.0;\nconst int m = 2;\noutp[i] = fin[i] * k + float(iin[i] * m);"
    ),
    # The input picks the element of a constant table.
    "constant-array": Template(
        "int k = iin[i] & 3;\noutp[i] = WEIGHTS[k] * fin[i] + WEIGHTS[3 - k] * 16.0;",
        "const float WEIGHTS[4] = {0.125, 0.25, 0.5, 1.0};",
    ),
    # The inner break ends only the inner loop, and the inner continue skips one k of every j.
    "nested-loop": Template(
        "float acc = 0.0;\nfor (int j = 0; j < 4; j++) {\n    for (int k = 0; k < 4; k++) {\n"
        "        if (k > j) {\n            break;\n        }\n        if (k == (iin[i] & 3)) {\n"
        "            continue;\n        }\n        acc += float(j * 4 + k);\n    }\n}\noutp[i] = acc + fin[i];"
    ),
    # m[1] replaces column 1 and m[0].y one element of column 0.
    "matrix-column-assignment": Template(
        "$mat2 m = $mat2(1.0, 2.0, 3.0, 4.0);\nm[1] = $vec2(fin[i], 5.0);\nm[0].y = float(iin[i]);\n"
        "outp[i] = m[0][0] + m[0][1] * 10.0 + m[1][0] * 100.0 + m[1][1] * 1000.0;"
    ),
    # A break in a case ends the switch, and a continue in a case skips the rest of the loop body.
    "switch-in-loop": Template(
        "float acc = 0.0;\nfor (int k = 0; k < 6; k++) {\n    switch ((iin[i] + k) & 3) {\n"
        "        case 0:\n            acc += 1.0;\n            break;\n        case 1:\n            continue;\n"
        "        case 2:\n            acc += fin[i];\n            break;\n        default:\n"
        "            acc *= 2.0;\n            break;\n    }\n    acc += 0.5;\n}\noutp[i] = acc;"
    ),
    # A continue in a do-while loop still tests the condition; the break only bounds a translation that skips it.
    "do-while-continue": Template(
        "int m = 5 + (iin[i] & 3);\nint k = 0;\nfloat acc = fin[i];\ndo {\n    k++;\n    if (k > 20) {\n"
        "        break;\n    }\n    if (k > 4) {\n        acc += 0.5;\n        continue;\n    }\n"
        "    acc += float(k);\n} while (k < m);\noutp[i] = acc;"
    ),
    # ?: groups to the right, and its last operand takes the whole sum. Scaling by a power of two is exact, so
    # contracting a product into a fused multiply-add does not change the result.
    "chained-conditional": Template(
        "int a = iin[i];\nfloat b = a > 1 ? a > 9 ? 1.0 : 2.0 : 3.0;\nfloat c = a > 6 ? 4.0 : 5.0 + fin[i];\n"
        "outp[i] = (a < 4 ? fin[i] : a < 8 ? fin[i] * 2.0 : a < 12 ? 3.0 : -fin[i]) + b * 10.0 + c * 256.0;"
    ),
    # No parentheses: each operator binds as C does, and - and / group to the left.
    "operator-precedence": Template(
        "int a = iin[i];\nint r = a + 3 << 2 & 60 ^ a * 2 | 1;\nint s = 20 - a - 3 + a / 2 * 3;\n"
        "outp[i] = float(r) + float(s) * 64.0 + fin[i] / 2.0 * 4.0 - 1.0 - 2.0;"
    ),
    "matrix-arithmetic": Template(
        "$mat2 m = $mat2(fin[i], 1.0, 2.0, 3.0);\n$mat2 n = $mat2(4.0, float(iin[i]), 5.0, 6.0);\n"
        "$mat2 r = (m + n) * 2.0 - 0.5 * n;\nr += m;\n"
        "outp[i] = r[0][0] + r[0][1] * 10.0 + r[1][0] * 100.0 + r[1][1] * 1000.0;"
    ),
    "vector-array": Template(
        "$vec2 pts[3];\nfor (int k = 0; k < 3; k++) {\n    pts[k] = $vec2(fin[i] + float(k), float(iin[i] * k));\n}\n"
        "int j = iin[i] % 3;\npts[j].y = pts[j].y + 100.0;\n"
        "outp[i] = pts[0].x + pts[1].y * 10.0 + pts[2].x * 1024.0 + pts[j].y;"
    ),
    # A break leaves only the innermost switch, and case 1 of the second inner switch falls through into case 2.
    "nested-switch": Template(
        "float acc = 0.0;\nint a = iin[i];\nswitch (a & 3) {\n    case 0:\n        switch ((a >> 2) & 1) {\n"
        "            case 0:\n                acc += 1.0;\n                break;\n            default:\n"
        "                acc += 2.0;\n                break;\n        }\n        acc += 4.0;\n        break;\n"
        "    case 1:\n        switch (a >> 2) {\n            case 1:\n                acc += 8.0;\n"
        "            case 2:\n                acc += 16.0;\n                break;\n        }\n        acc += 32.0;\n"
        "        break;\n    default:\n        acc += 64.0;\n        break;\n}\noutp[i] = acc + fin[i];"
    ),
    # The selector is unsigned, and two labels lie above 2^31, where a signed reading of them would be negative.
    "unsigned-switch": Template(
        "uint u = uint(iin[i] - 8);\nfloat r = 0.0;\nswitch (u) {\n    case 0u:\n        r = 1.0;\n        break;\n"
        "    case 4294967295u:\n        r = 2.0;\n        break;\n    case 4294967288u:\n        r = 3.0;\n"
        "        break;\n    case 7u:\n        r = 4.0;\n        break;\n    default:\n"
        "        r = float(u % 5u) * 8.0;\n        break;\n}\noutp[i] = r + fin[i];"
    ),
    # Both counters step after a continue too.
    "multi-variable-loop": Template(
        "float acc = 0.0;\nint n = 6 + (iin[i] & 3);\nfor (int a = 0, b = n; a < b; a++, b--) {\n"
        "    if (a == 1) {\n        continue;\n    }\n    acc += float(a * 16 + b);\n}\noutp[i] = acc + fin[i];"
    ),
    "forward-declaration": Template(
        "outp[i] = quadruple(fin[i]) + quadruple(float(iin[i]));", QUAD + TWICE
    ),
    # m has two columns of three rows: m * v takes two components and gives three, v * m the reverse, and m[1] is
    # the second column. The entries are small integers, so every product and sum is exact.
    "non-square-matrix": Template(
        "float a = float(iin[i]);\n$mat2x3 m = $mat2x3(a, 1.0, 2.0, 3.0, 4.0, 5.0);\n"
        "$vec3 v = m * $vec2(1.0, 2.0);\n$vec2 w = $vec3(1.0, 1.0, 2.0) * m;\n"
        "outp[i] = v.x + v.z * 16.0 + w.x * 256.0 + w.y * 4096.0 + m[1].z * 65536.0;"
    ),
    # Swizzles read and write vector members; the weights keep the sum below 2^24, so it is exact.
    "struct-vector-member": Template(
        "Pose p;\np.v = $vec4(float(iin[i]), 1.0, 2.0, 3.0);\np.u = p.v.wx;\np.n = iin[i] & 3;\n"
        "p.v.yz = p.u * 2.0;\np.u.y = p.v.y + 1.0;\n"
        "outp[i] = p.v.x + p.v.y * 16.0 + p.v.z * 128.0 + p.v.w * 4096.0 + p.u.x * 16384.0 + p.u.y * 65536.0"
        " + float(p.n) * 1048576.0;",
        POSE,
    ),
    # Only the operand that the condition picks is evaluated, so k changes once per conditional expression.
    "conditional-side-effect": Template(
        "int k = iin[i];\nfloat r = (k & 1) == 0 ? bump(k) : leap(k);\nfloat s = k > 20 ? leap(k) : 0.5;\n"
        "outp[i] = r + float(k) * 64.0 + s * 4096.0;",
        STEPS,
    ),
    "switch-return": Template(
        "outp[i] = pick(iin[i]) + pick(iin[i] + 1) * 128.0;", PICK
    ),
    # A break in a loop inside a case leaves only the loop, and a continue there starts its next iteration.
    "loop-in-switch": Template(
        "float acc = 0.0;\nint n = iin[i];\nswitch (n & 3) {\n    case 0:\n        for (int k = 0; k < 8; k++) {\n"
        "            if (k == (n >> 2)) {\n                break;\n            }\n            acc += 2.0;\n        }\n"
        "        acc += 1.0;\n        break;\n    case 1:\n        while (acc < 20.0) {\n            acc += 3.0;\n"
        "            if (acc > float(n)) {\n                break;\n            }\n        }\n        acc += 128.0;\n"
        "        break;\n    default:\n        for (int k = 0; k < n; k++) {\n            if ((k & 1) == 1) {\n"
        "                continue;\n            }\n            acc += 4.0;\n        }\n        acc += 512.0;\n"
        "        break;\n}\noutp[i] = acc + fin[i];"
    ),
    # The int and uint operands and the integer literals convert to float without a constructor.
    "implicit-conversion": Template(
        "int a = iin[i];\nuint u = 3u;\nfloat x = a;\nfloat y = a * 0.5 + 1;\nfloat z = x / 4 + u;\n"
        "outp[i] = x + y * 16.0 + z * 256.0;"
    ),
    # m *= n is the matrix product m * n, and v *= m the row-vector product v * m; neither is element-wise.
    "matrix-compound-assignment": Template(
        "float f = float(iin[i]);\n$mat2 m = $mat2(f, 1.0, 2.0, 3.0);\n$mat2 n = $mat2(1.0, 2.0, 0.0, 1.0);\n"
        "m *= n;\nm *= 2.0;\nm -= n;\n$vec2 v = $vec2(1.0, 1.0);\nv *= m;\n"
        "outp[i] = m[0][0] + m[0][1] * 16.0 + m[1][0] * 256.0 + m[1][1] * 4096.0 + v.x * 65536.0"
        " + v.y * 1048576.0;"
    ),
    # A matrix built from one scalar has the scalar on its diagonal and zeros elsewhere.
    "diagonal-matrix": Template(
        "float f = float(iin[i]);\n$mat2 d = $mat2(f);\n$mat2 e = $mat2(2.0) + d;\n"
        "outp[i] = d[0][0] + d[0][1] * 16.0 + d[1][0] * 256.0 + d[1][1] * 4096.0 + e[1][1] * 65536.0"
        " + e[0][1] * 1048576.0;"
    ),
    # Both loops have no condition of their own and end only through their break.
    "infinite-loop": Template(
        "int k = iin[i];\nfloat acc = 0.0;\nfor (;;) {\n    acc += 2.0;\n    k = k + 3;\n    if (k > 20) {\n"
        "        break;\n    }\n}\nwhile (true) {\n    acc += 64.0;\n    if (acc > 200.0) {\n        break;\n"
        "    }\n}\noutp[i] = acc + float(k) * 1024.0;"
    ),
    # A loop condition is evaluated before every iteration, and && skips its right operand when the left is false.
    "loop-condition-call": Template(
        "int k = iin[i];\nfloat acc = 0.0;\nwhile (advance(k) < 40) {\n    acc += 1.0;\n}\n"
        "for (int j = advance(k); j < 64 && advance(k) < 60; j += 4) {\n    acc += 32.0;\n}\n"
        "outp[i] = acc + float(k) * 4096.0;",
        ADVANCE,
    ),
    # A compound assignment evaluates its left operand once, so each one calls slot once.
    "indexed-compound-assignment": Template(
        "float a[4];\nfor (int j = 0; j < 4; j++) {\n    a[j] = float(j);\n}\nint k = iin[i] & 1;\n"
        "a[slot(k)] += 8.0;\na[slot(k)] *= 4.0;\na[k - 1] -= 0.5;\n"
        "outp[i] = a[0] + a[1] * 16.0 + a[2] * 256.0 + a[3] * 4096.0 + float(k) * 65536.0;",
        SLOT,
    ),
    # Vector compound assignments work component by component, with a scalar operand applied to every component.
    "vector-compound-assignment": Template(
        "$vec4 v = $vec4(float(iin[i]), 1.0, 2.0, 3.0);\n$vec4 w = $vec4(0.5, 0.25, 4.0, 1.0);\n"
        "v += w;\nv *= 2.0;\nv -= w * 2.0;\nv /= $vec4(2.0, 4.0, 8.0, 2.0);\nv *= w;\nv.y += 1.0;\n"
        "outp[i] = v.x + v.y * 64.0 + v.z * 4096.0 + v.w * 65536.0;"
    ),
    # Stores through a member of an array element of a struct, with indices the input picks.
    "nested-member-assignment": Template(
        "Bag b;\nfor (int j = 0; j < 2; j++) {\n    b.items[j].pos.x = float(j);\n"
        "    b.items[j].pos.y = float(j) + 2.0;\n    b.items[j].w[0] = 1.0;\n    b.items[j].w[1] = 3.0;\n}\n"
        "int k = iin[i] & 1;\nb.n = iin[i];\nb.items[k].pos.y = float(b.n) * 2.0;\nb.items[1 - k].w[k] += 8.0;\n"
        f"b.items[k].w[1 - k] = b.items[1 - k].pos.x + 4.0;\noutp[i] = {BAG_SUM};",
        BAG,
    ),
    "bool-array": Template(
        "bool flags[4];\nfor (int j = 0; j < 4; j++) {\n    flags[j] = ((iin[i] >> j) & 1) == 1;\n}\n"
        "bool either = flags[0] || flags[3];\nflags[1] = !flags[1];\n"
        "outp[i] = (flags[1] ? 1.0 : 0.0) + (flags[2] ? 2.0 : 0.0) + (either ? 4.0 : 0.0) + float(iin[i]) * 8.0;"
    ),
    "compute": Template("outp[i] = fin[i];"),
    **OPERATORS,
}


def scalar(spelling: str, source: str = "iin", prelude: str = "") -> Template:
    return Template(
        f"{spelling} v = {spelling}({source}[i]);\noutp[i] = float(v * v);",
        prelude=prelude,
    )
