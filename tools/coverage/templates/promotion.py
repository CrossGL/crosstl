"""Integer promotion probe: two signed 8-bit integers multiply as ``int``.

CrossGL, C and C++ convert 8- and 16-bit integer operands to ``int`` before they multiply them (C11 6.3.1.1, C++
[conv.prom]), and MSL follows C++ here. The probe squares an 8-bit input and outputs the product without converting
it back. The inputs run from 0 to 15, so from 12 up the product leaves the 8-bit range, and a translation that
multiplies at 8 bits writes the wrapped product instead: -112 for 12 rather than 144. The 8- and 16-bit type probes
(:func:`.base.narrow_scalar`) check the wrapped product, which every language computes the same way.

GLSL (``GL_EXT_shader_explicit_arithmetic_types_int8``), Slang, Rust and Mojo multiply at the operands' width, so
their probes convert the operands to a 32-bit integer first. HLSL has no 8-bit arithmetic type (``int8_t4_packed``
only packs four bytes into a 32-bit integer), and GLSL ES 3.00 and WGSL have no 8-bit integers.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("integer-promotion",)

_WIDENED = "int8_t v = int8_t(iin[i]);\noutp[i] = float(int(v) * int(v));"

CROSSGL = {
    "integer-promotion": Template("int8_t v = int8_t(iin[i]);\noutp[i] = float(v * v);")
}
GLSL = {
    "integer-promotion": Template(
        _WIDENED,
        prelude="#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require",
    )
}
SLANG = {"integer-promotion": Template(_WIDENED)}
KERNELS = {
    "integer-promotion": Template(
        "int8_t v = int8_t(iin[i]);\noutp[i] = float(v * v);",
        prelude="#include <cstdint>",
    )
}
METAL = {
    "integer-promotion": Template("char v = char(iin[i]);\noutp[i] = float(v * v);")
}
OPENCL = {
    "integer-promotion": Template("char v = (char)iin[i];\noutp[i] = (float)(v * v);")
}
RUST = {
    "integer-promotion": Template(
        "let v: i8 = iin[i] as i8;\noutp[i] = ((v as i32) * (v as i32)) as f32;"
    )
}
MOJO = {
    "integer-promotion": Template(
        "var v: Int8 = Int8(iin[i])\n"
        "outp[i] = Float32(v.cast[DType.int32]() * v.cast[DType.int32]())"
    )
}

ELSEWHERE = dict.fromkeys(NAMES, ABSENT)
