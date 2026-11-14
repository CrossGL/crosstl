"""Float-remainder, logical-xor, constant-array-size, increment-in-index, empty-statement and default-only-switch
probes: their C-family form (:data:`FAMILY`, as HLSL and CrossGL write them) and the languages that write them
differently.

The remainder is the one C's ``fmod`` computes, whose sign is the dividend's. GLSL has no remainder on floats (its
``mod`` is floored), so its probe computes ``x - y * trunc(x / y)``, as Mojo's does, whose ``%`` on floats is floored
too. Metal and OpenCL C call ``fmod``, CUDA ``fmodf``. No quotient in the probe is a whole number, so ``trunc`` gives
the same remainder ``fmod`` does on every runtime.

Only GLSL has a logical exclusive or, ``^^``; the other languages compare two ``bool`` values with ``!=``, which the
boolean-logic probe measures, so their logical-xor probe is absent. Only the C family writes an increment inside an
expression; WGSL increments are statements and Rust and Mojo have none, so increment-in-index is absent there. Mojo
has no ``switch``.
"""

from __future__ import annotations

from .base import ABSENT, Template
from .statements import c_text

_REMAINDER = (
    "float a = float(iin[i]) - 7.5;\nfloat b = a % 2.0;\nfloat c = a % -3.0;\n"
    "$vec2 v = $vec2(a, -a) % $vec2(4.0, -2.0);\nfloat d = a * 3.0;\nd %= 4.0;\n"
    "outp[i] = b + c * 8.0 + v.x * 64.0 + v.y * 512.0 + d * 4096.0;"
)
# GLSL and Mojo: the same remainders through trunc.
_TRUNCATED = (
    "float a = float(iin[i]) - 7.5;\nfloat b = a - 2.0 * trunc(a / 2.0);\nfloat c = a + 3.0 * trunc(a / -3.0);\n"
    "$vec2 w = $vec2(a, -a);\n$vec2 v = w - $vec2(4.0, -2.0) * trunc(w / $vec2(4.0, -2.0));\n"
    "float d = a * 3.0;\nd = d - 4.0 * trunc(d / 4.0);\n"
    "outp[i] = b + c * 8.0 + v.x * 64.0 + v.y * 512.0 + d * 4096.0;"
)
_XOR = (
    "bool p = iin[i] > 7;\nbool q = (iin[i] & 1) == 1;\nbool r = (p ^^ q) ^^ (iin[i] == 3);\n"
    "outp[i] = (p ^^ q ? 1.0 : 0.0) + (r ? 2.0 : 0.0) + (!(p ^^ true) ? 4.0 : 0.0);"
)
_SIZE_SUM = "t[N - 1] + s[N * 2 - 1] * 32.0 + s[{0}] * 8192.0"
_SIZED = (
    "float t[N];\nfor (int k = 0; k < N; k++) {\n    t[k] = float(iin[i] + k);\n}\nfloat s[N * 2];\n"
    "for (int k = 0; k < N * 2; k++) {\n    s[k] = t[k % N] * float(k);\n}\n"
    f"outp[i] = {_SIZE_SUM.format('iin[i] & 7')};"
)
_INCREMENT = (
    "float t[4];\nint k = 0;\nt[k++] = float(iin[i]);\nt[k++] = 2.0;\nt[k] = 3.0;\nt[++k] = 4.0;\n"
    "float a = t[k--];\nfloat b = t[--k];\n"
    "outp[i] = t[0] + a * 32.0 + b * 1024.0 + float(k) * 32768.0 + t[2] * 65536.0;"
)
_EMPTY = (
    "float r = float(iin[i]);\n;\n{}\nint k = 0;\nfor (; k < iin[i]; k++) {}\nif (iin[i] > 7) {} else {\n"
    "    r += 32.0;\n}\nwhile (k > 20) {}\noutp[i] = r + float(k) * 64.0;"
)
# A switch with only a default label, and one whose first case is empty.
_DEFAULT_ONLY = (
    "float r = 1.0;\nswitch (iin[i]) {\n    default:\n        r += float(iin[i]) * 2.0;\n        break;\n}\n"
    "switch (iin[i] & 3) {\n    case 0:\n        break;\n    default:\n        r += 32.0;\n        break;\n}\n"
    "outp[i] = r;"
)

FAMILY = {
    "float-remainder": Template(_REMAINDER),
    "logical-xor": ABSENT,
    "constant-array-size": Template(_SIZED, "static const int N = 4;"),
    "increment-in-index": Template(_INCREMENT),
    "empty-statement": Template(_EMPTY),
    "default-only-switch": Template(_DEFAULT_ONLY),
}

GLSL = {
    "float-remainder": Template(_TRUNCATED),
    "logical-xor": Template(_XOR),
    "constant-array-size": Template(_SIZED, "const int N = 4;"),
}

METAL = {
    "float-remainder": Template(
        "float a = float(iin[i]) - 7.5;\nfloat b = fmod(a, 2.0);\nfloat c = fmod(a, -3.0);\n"
        "float2 v = fmod(float2(a, -a), float2(4.0, -2.0));\nfloat d = a * 3.0;\nd = fmod(d, 4.0);\n"
        "outp[i] = b + c * 8.0 + v.x * 64.0 + v.y * 512.0 + d * 4096.0;"
    ),
    "constant-array-size": Template(_SIZED, "constant int N = 4;"),
}


def _c_family(unsigned: str, constant: str, remainder: str) -> dict[str, Template]:
    """The C family's probes as CUDA and HIP (``constant`` ``constexpr int N = 4;``) and OpenCL C write them.

    ``remainder`` is the probe's float-remainder body: CUDA has no vector arithmetic, so it takes each component's
    remainder with ``fmodf``.
    """
    return {
        "float-remainder": Template(remainder),
        "logical-xor": ABSENT,
        "constant-array-size": Template(c_text(_SIZED, unsigned), constant),
        "increment-in-index": Template(c_text(_INCREMENT, unsigned)),
        "empty-statement": Template(c_text(_EMPTY, unsigned)),
        "default-only-switch": Template(c_text(_DEFAULT_ONLY, unsigned)),
    }


CUDA = _c_family(
    "unsigned int",
    "constexpr int N = 4;",
    "float a = (float)(iin[i]) - 7.5f;\nfloat b = fmodf(a, 2.0f);\nfloat c = fmodf(a, -3.0f);\n"
    "float vx = fmodf(a, 4.0f);\nfloat vy = fmodf(-a, -2.0f);\nfloat d = a * 3.0f;\nd = fmodf(d, 4.0f);\n"
    "outp[i] = b + c * 8.0f + vx * 64.0f + vy * 512.0f + d * 4096.0f;",
)
OPENCL = _c_family(
    "uint",
    "#define N 4",
    "float a = (float)(iin[i]) - 7.5f;\nfloat b = fmod(a, 2.0f);\nfloat c = fmod(a, -3.0f);\n"
    "float2 v = fmod((float2)(a, -a), (float2)(4.0f, -2.0f));\nfloat d = a * 3.0f;\nd = fmod(d, 4.0f);\n"
    "outp[i] = b + c * 8.0f + v.x * 64.0f + v.y * 512.0f + d * 4096.0f;",
)

WGSL = {
    "float-remainder": Template(
        "let a = f32(iin[i]) - 7.5;\nlet b = a % 2.0;\nlet c = a % -3.0;\n"
        "let v = vec2<f32>(a, -a) % vec2<f32>(4.0, -2.0);\nvar d = a * 3.0;\nd %= 4.0;\n"
        "outp[i] = b + c * 8.0 + v.x * 64.0 + v.y * 512.0 + d * 4096.0;"
    ),
    "logical-xor": ABSENT,
    "constant-array-size": Template(
        "var t: array<f32, N>;\nfor (var k = 0; k < N; k++) {\n    t[k] = f32(iin[i] + k);\n}\n"
        "var s: array<f32, N * 2>;\nfor (var k = 0; k < N * 2; k++) {\n    s[k] = t[k % N] * f32(k);\n}\n"
        f"outp[i] = {_SIZE_SUM.format('iin[i] & 7')};",
        "const N = 4;",
    ),
    "increment-in-index": ABSENT,
    "empty-statement": Template(
        "var r = f32(iin[i]);\n;\n{}\nvar k = 0;\nfor (; k < iin[i]; k++) {}\nif (iin[i] > 7) {} else {\n"
        "    r += 32.0;\n}\nwhile (k > 20) {}\noutp[i] = r + f32(k) * 64.0;"
    ),
    "default-only-switch": Template(
        "var r: f32 = 1.0;\nswitch (iin[i]) {\n    default: {\n        r += f32(iin[i]) * 2.0;\n    }\n}\n"
        "switch (iin[i] & 3) {\n    case 0: {\n    }\n    default: {\n        r += 32.0;\n    }\n}\noutp[i] = r;"
    ),
}

RUST = {
    "float-remainder": Template(
        "let a = iin[i] as f32 - 7.5;\nlet b = a % 2.0;\nlet c = a % -3.0;\n"
        "let v = $vec2::new(a % 4.0, -a % -2.0);\nlet mut d = a * 3.0;\nd %= 4.0;\n"
        "outp[i] = b + c * 8.0 + v.x * 64.0 + v.y * 512.0 + d * 4096.0;"
    ),
    "logical-xor": ABSENT,
    "constant-array-size": Template(
        "let mut t = [0.0f32; N];\nfor k in 0..N {\n    t[k] = (iin[i] + k as i32) as f32;\n}\n"
        "let mut s = [0.0f32; N * 2];\nfor k in 0..N * 2 {\n    s[k] = t[k % N] * k as f32;\n}\n"
        f"outp[i] = {_SIZE_SUM.format('(iin[i] & 7) as usize')};",
        "const N: usize = 4;",
    ),
    "increment-in-index": ABSENT,
    "empty-statement": Template(
        "let mut r = iin[i] as f32;\n;\n{}\nlet mut k = 0;\nwhile k < iin[i] {\n    k += 1;\n}\n"
        "if iin[i] > 7 {} else {\n    r += 32.0;\n}\nwhile k > 20 {}\noutp[i] = r + k as f32 * 64.0;"
    ),
    "default-only-switch": Template(
        "let mut r = 1.0;\nmatch iin[i] {\n    _ => {\n        r += iin[i] as f32 * 2.0;\n    }\n}\n"
        "match iin[i] & 3 {\n    0 => {}\n    _ => {\n        r += 32.0;\n    }\n}\noutp[i] = r;"
    ),
}

MOJO = {
    "float-remainder": Template(
        "var a = Float32(iin[i]) - 7.5\nvar b = a - 2.0 * trunc(a / 2.0)\nvar c = a + 3.0 * trunc(a / -3.0)\n"
        "var w = $vec2(a, -a)\nvar v = w - $vec2(4.0, -2.0) * trunc(w / $vec2(4.0, -2.0))\n"
        "var d = a * 3.0\nd = d - 4.0 * trunc(d / 4.0)\n"
        "outp[i] = b + c * 8.0 + v[0] * 64.0 + v[1] * 512.0 + d * 4096.0",
        prelude="from std.math import trunc",
    ),
    "logical-xor": ABSENT,
    "constant-array-size": Template(
        "var t = Array[Float32, N](fill=0.0)\nfor k in range(N):\n    t[k] = Float32(iin[i] + Int32(k))\n"
        "var s = Array[Float32, N * 2](fill=0.0)\nfor k in range(N * 2):\n    s[k] = t[k % N] * Float32(k)\n"
        f"outp[i] = {_SIZE_SUM.format('Int(iin[i] & 7)')}",
        "comptime N = 4",
    ),
    "increment-in-index": ABSENT,
    "empty-statement": Template(
        "var r = Float32(iin[i])\npass\nvar k: Int32 = 0\nwhile k < iin[i]:\n    k += 1\nif iin[i] > 7:\n"
        "    pass\nelse:\n    r += 32.0\nwhile k > 20:\n    pass\noutp[i] = r + Float32(k) * 64.0"
    ),
    "default-only-switch": ABSENT,
}
