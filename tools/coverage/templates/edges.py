"""Loop-counter-write, unreachable-statement, float-loop-counter, signed-division, matrix-negation and
struct-array-copy probes: their C-family form (:data:`FAMILY`) and the languages that write them differently.

The C family divides integers towards zero and shifts signed integers arithmetically, as WGSL and Rust do; Mojo's
``//`` rounds down, so its probe divides through a helper that rounds towards zero. Rust and Mojo have no C-style
``for``, so their probes count with ``while``, which keeps a write to the counter inside the body. Metal's and WGSL's
unary minus take no matrices, so their probes multiply by -1. The matrices are symmetric, so the probes read the same
whether a language indexes a matrix by columns or by rows. CUDA, HIP and OpenCL C cannot assign arrays, and CUDA,
HIP, OpenCL C and Mojo have no matrices.
"""

from __future__ import annotations

from .base import ABSENT, PAIR, Template
from .statements import c_text

# cap has a statement after its return, as the loop in the unreachable-statement probe has after break and continue.
CAP = "float cap(float x) {\n    if (x > 1.0) {\n        return 1.0;\n        x = 0.0;\n    }\n    return x;\n}"

_COUNTER_SUM = "acc + {0}(n) * 128.0"
_FLOAT_SUM = "acc + {0}(n) * 1024.0"
_DIVISION_SUM = "{0}(q) + {0}(h) * 16.0 + {0}(s) * 256.0 + {0}(t) * 4096.0"
_NEGATION_SUM = "n{c0}{r0} + n{c0}{r1} * 16.0 + n{c1}{r1} * 256.0 + p{c1}{r1} * 4096.0 + p{c0}{r0} * 65536.0"
_COPY_SUM = (
    "a[0].a + a[j].b * 16.0 + b[j].b * 256.0 + b[0].a * 4096.0 + b[2].b * 65536.0"
)
_INDEXED = {"c0": "[0]", "c1": "[1]", "r0": "[0]", "r1": "[1]"}
_AXES = {"c0": ".x_axis", "c1": ".y_axis", "r0": ".x", "r1": ".y"}

FAMILY = {
    # The body moves the counter forward in one loop and back in the other; the increment still runs after each.
    "loop-counter-write": Template(
        "int j = iin[i] & 7;\nfloat acc = 0.0;\nfor (int k = 0; k < 12; k++) {\n    if (k == j) {\n"
        "        k += 3;\n    }\n    acc += float(k);\n}\nint n = 0;\nfor (int m = 0; m < 6; m++) {\n    n++;\n"
        "    if (m == (iin[i] & 3) && n < 5) {\n        m--;\n    }\n}\n"
        f"outp[i] = {_COUNTER_SUM.format('float')};"
    ),
    "unreachable-statement": Template(
        "float acc = 0.0;\nfor (int k = 0; k < 8; k++) {\n    if (k == iin[i]) {\n        break;\n"
        "        acc += 100.0;\n    }\n    if ((k & 1) == 1) {\n        continue;\n        acc += 50.0;\n    }\n"
        "    acc += 1.0;\n}\noutp[i] = acc + cap(fin[i]) * 16.0;",
        CAP,
    ),
    # Every counter value is a multiple of a power of two, so each sum is exact.
    "float-loop-counter": Template(
        "float acc = 0.0;\nint n = 0;\nfor (float t = 0.0; t < fin[i] + 4.0; t += 0.25) {\n    acc += t;\n"
        "    n++;\n}\nfor (float s = 2.0; s > 0.0; s -= 0.5) {\n    acc += s * 64.0;\n}\n"
        f"outp[i] = {_FLOAT_SUM.format('float')};"
    ),
    # a runs from -8 to 7, so quotients round towards zero and the shift rounds down.
    "signed-division": Template(
        "int a = iin[i] - 8;\nint b = (iin[i] & 1) == 0 ? -3 : 5;\nint q = a / b;\nint h = a / 2;\n"
        f"int s = a >> 1;\nint t = (a * 5) / -4;\noutp[i] = {_DIVISION_SUM.format('float')};"
    ),
    "matrix-negation": Template(
        "float f = float(iin[i]);\n$mat2 m = $mat2(f, 1.0, 1.0, 3.0);\n$mat2 n = -m;\n$mat2 p = -(m * 2.0) + n;\n"
        f"outp[i] = {_NEGATION_SUM.format(**_INDEXED)};"
    ),
    # Assigning the array copies its structs, so the writes after it reach only one of the two arrays.
    "struct-array-copy": Template(
        "Pair a[3];\nfor (int k = 0; k < 3; k++) {\n    a[k].a = float(k);\n    a[k].b = fin[i];\n}\n"
        "Pair b[3] = a;\nint j = iin[i] % 3;\nb[j].b = 7.0;\na[0].a = 0.5;\n"
        f"outp[i] = {_COPY_SUM};",
        PAIR,
    ),
}

METAL = {
    "matrix-negation": Template(
        "float f = float(iin[i]);\n$mat2 m = $mat2(f, 1.0, 1.0, 3.0);\n$mat2 n = m * -1.0;\n"
        f"$mat2 p = (m * 2.0) * -1.0 + n;\noutp[i] = {_NEGATION_SUM.format(**_INDEXED)};"
    ),
    "struct-array-copy": Template(
        FAMILY["struct-array-copy"]
        .body.replace("Pair a[3];", "array<Pair, 3> a;")
        .replace("Pair b[3] = a;", "array<Pair, 3> b = a;"),
        PAIR,
    ),
}


def _c_family(unsigned: str, device: str) -> dict[str, Template]:
    """The C family's probes as CUDA, HIP (``device`` ``__device__ ``) and OpenCL C (``device`` empty) write them."""
    return {
        "loop-counter-write": Template(
            c_text(FAMILY["loop-counter-write"].body, unsigned)
        ),
        "unreachable-statement": Template(
            c_text(FAMILY["unreachable-statement"].body, unsigned),
            device + c_text(CAP, unsigned),
        ),
        "float-loop-counter": Template(
            c_text(FAMILY["float-loop-counter"].body, unsigned)
        ),
        "signed-division": Template(c_text(FAMILY["signed-division"].body, unsigned)),
        "matrix-negation": ABSENT,
        "struct-array-copy": ABSENT,
    }


CUDA = _c_family("unsigned int", "__device__ ")
OPENCL = _c_family("uint", "")

WGSL = {
    "loop-counter-write": Template(
        "let j = iin[i] & 7;\nvar acc: f32 = 0.0;\nfor (var k = 0; k < 12; k++) {\n    if (k == j) {\n"
        "        k += 3;\n    }\n    acc += f32(k);\n}\nvar n = 0;\nfor (var m = 0; m < 6; m++) {\n    n++;\n"
        "    if (m == (iin[i] & 3) && n < 5) {\n        m--;\n    }\n}\n"
        f"outp[i] = {_COUNTER_SUM.format('f32')};"
    ),
    "unreachable-statement": Template(
        "var acc: f32 = 0.0;\nfor (var k = 0; k < 8; k++) {\n    if (k == iin[i]) {\n        break;\n"
        "        acc += 100.0;\n    }\n    if ((k & 1) == 1) {\n        continue;\n        acc += 50.0;\n    }\n"
        "    acc += 1.0;\n}\noutp[i] = acc + cap(fin[i]) * 16.0;",
        "fn cap(x_in: f32) -> f32 {\n    var x = x_in;\n    if (x > 1.0) {\n        return 1.0;\n        x = 0.0;\n"
        "    }\n    return x;\n}",
    ),
    "float-loop-counter": Template(
        "var acc: f32 = 0.0;\nvar n = 0;\nfor (var t: f32 = 0.0; t < fin[i] + 4.0; t += 0.25) {\n    acc += t;\n"
        "    n++;\n}\nfor (var s: f32 = 2.0; s > 0.0; s -= 0.5) {\n    acc += s * 64.0;\n}\n"
        f"outp[i] = {_FLOAT_SUM.format('f32')};"
    ),
    "signed-division": Template(
        "let a = iin[i] - 8;\nlet b = select(5, -3, (iin[i] & 1) == 0);\nlet q = a / b;\nlet h = a / 2;\n"
        f"let s = a >> 1u;\nlet t = (a * 5) / -4;\noutp[i] = {_DIVISION_SUM.format('f32')};"
    ),
    "matrix-negation": Template(
        "let f = f32(iin[i]);\nlet m = mat2x2<f32>(f, 1.0, 1.0, 3.0);\nlet n = m * -1.0;\n"
        f"let p = (m * 2.0) * -1.0 + n;\noutp[i] = {_NEGATION_SUM.format(**_INDEXED)};"
    ),
    "struct-array-copy": Template(
        "var a: array<Pair, 3>;\nfor (var k = 0; k < 3; k++) {\n    a[k].a = f32(k);\n    a[k].b = fin[i];\n}\n"
        "var b = a;\nlet j = iin[i] % 3;\nb[j].b = 7.0;\na[0].a = 0.5;\n"
        f"outp[i] = {_COPY_SUM};",
        "struct Pair {\n    a: f32,\n    b: f32,\n}",
    ),
}

_RUST_COUNTERS = (
    "let j = iin[i] & 7;\nlet mut acc: f32 = 0.0;\nlet mut k = 0;\nwhile k < 12 {\n    if k == j {\n        k += 3;\n"
    "    }\n    acc += k as f32;\n    k += 1;\n}\nlet mut n = 0;\nlet mut m = 0;\nwhile m < 6 {\n    n += 1;\n"
    "    if m == (iin[i] & 3) && n < 5 {\n        m -= 1;\n    }\n    m += 1;\n}\noutp[i] = acc + n as f32 * 128.0;"
)

RUST = {
    "loop-counter-write": Template(_RUST_COUNTERS),
    "unreachable-statement": Template(
        "let mut acc: f32 = 0.0;\nfor k in 0..8 {\n    if k == iin[i] {\n        break;\n        acc += 100.0;\n"
        "    }\n    if (k & 1) == 1 {\n        continue;\n        acc += 50.0;\n    }\n    acc += 1.0;\n}\n"
        "outp[i] = acc + cap(fin[i]) * 16.0;",
        "fn cap(mut x: f32) -> f32 {\n    if x > 1.0 {\n        return 1.0;\n        x = 0.0;\n    }\n    x\n}",
    ),
    "float-loop-counter": Template(
        "let mut acc: f32 = 0.0;\nlet mut n = 0;\nlet mut t: f32 = 0.0;\nwhile t < fin[i] + 4.0 {\n    acc += t;\n"
        "    n += 1;\n    t += 0.25;\n}\nlet mut s: f32 = 2.0;\nwhile s > 0.0 {\n    acc += s * 64.0;\n"
        "    s -= 0.5;\n}\noutp[i] = acc + n as f32 * 1024.0;"
    ),
    "signed-division": Template(
        "let a = iin[i] - 8;\nlet b = if (iin[i] & 1) == 0 { -3 } else { 5 };\nlet q = a / b;\nlet h = a / 2;\n"
        "let s = a >> 1;\nlet t = (a * 5) / -4;\n"
        "outp[i] = q as f32 + h as f32 * 16.0 + s as f32 * 256.0 + t as f32 * 4096.0;"
    ),
    "matrix-negation": Template(
        "let f = iin[i] as f32;\nlet m = $mat2::from_cols($vec2::new(f, 1.0), $vec2::new(1.0, 3.0));\n"
        f"let n = -m;\nlet p = -(m * 2.0) + n;\noutp[i] = {_NEGATION_SUM.format(**_AXES)};"
    ),
    "struct-array-copy": Template(
        "let mut a = [Pair::default(); 3];\nfor k in 0..3 {\n    a[k].a = k as f32;\n    a[k].b = fin[i];\n}\n"
        "let mut b = a;\nlet j = (iin[i] % 3) as usize;\nb[j].b = 7.0;\na[0].a = 0.5;\n"
        f"outp[i] = {_COPY_SUM};",
        "#[derive(Clone, Copy, Default)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}",
    ),
}

MOJO = {
    "loop-counter-write": Template(
        "var j = iin[i] & 7\nvar acc: Float32 = 0.0\nvar k: Int32 = 0\nwhile k < 12:\n    if k == j:\n"
        "        k += 3\n    acc += Float32(k)\n    k += 1\nvar n: Int32 = 0\nvar m: Int32 = 0\nwhile m < 6:\n"
        "    n += 1\n    if m == (iin[i] & 3) and n < 5:\n        m -= 1\n    m += 1\n"
        "outp[i] = acc + Float32(n) * 128.0"
    ),
    "unreachable-statement": Template(
        "var acc: Float32 = 0.0\nfor k in range(8):\n    if k == Int(iin[i]):\n        break\n        acc += 100.0\n"
        "    if (k & 1) == 1:\n        continue\n        acc += 50.0\n    acc += 1.0\n"
        "outp[i] = acc + cap(fin[i]) * 16.0",
        "def cap(var x: Float32) -> Float32:\n    if x > 1.0:\n        return 1.0\n        x = 0.0\n    return x",
    ),
    "float-loop-counter": Template(
        "var acc: Float32 = 0.0\nvar n: Int32 = 0\nvar t: Float32 = 0.0\nwhile t < fin[i] + 4.0:\n    acc += t\n"
        "    n += 1\n    t += 0.25\nvar s: Float32 = 2.0\nwhile s > 0.0:\n    acc += s * 64.0\n    s -= 0.5\n"
        "outp[i] = acc + Float32(n) * 1024.0"
    ),
    "signed-division": Template(
        "var a = iin[i] - 8\nvar b: Int32 = -3 if (iin[i] & 1) == 0 else 5\nvar q = quotient(a, b)\n"
        "var h = quotient(a, 2)\nvar s = a >> 1\nvar t = quotient(a * 5, -4)\n"
        f"outp[i] = {_DIVISION_SUM.format('Float32')}",
        "def quotient(a: Int32, b: Int32) -> Int32:\n    var q = a // b\n    if q * b != a and (a < 0) != (b < 0):\n"
        "        q += 1\n    return q",
    ),
    "matrix-negation": ABSENT,
    "struct-array-copy": Template(
        "var a = Array[Pair, 3](fill=Pair(0.0, 0.0))\nfor k in range(3):\n    a[k].a = Float32(k)\n"
        "    a[k].b = fin[i]\nvar b = a.copy()\nvar j = Int(iin[i] % 3)\nb[j].b = 7.0\na[0].a = 0.5\n"
        f"outp[i] = {_COPY_SUM}",
        "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32",
    ),
}
