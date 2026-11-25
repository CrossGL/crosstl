"""Union probe: a local union of a float and a 32-bit unsigned integer, written through one member and read through
the other.

The probe stores its input in the float member, reads the bits back through the integer member, writes the top 16 of
them under the exponent of 1.0 through the integer member, and outputs the float member. Every result is a float in
[1, 2) that the input's bits decide, so a translation that stores the members apart, where reading the integer returns
zero or an undefined value and the float keeps its input, computes something else.

CUDA, HIP and Metal declare the union as C++ does, and OpenCL C as C99 does, which reads another member as the bytes
of the one written last (C99 6.5.2.3, footnote 82; OpenCL C 3.0 is based on it). Rust declares a ``#[repr(C)]``
union whose fields share offset 0 and reads a field in an ``unsafe`` block (The Rust Reference, "Unions"). CrossGL
marks a struct ``@union_layout(size, alignment, byte order, ABI)`` and each member ``@union_member_layout(offset, size,
alignment)``, as the Metal and CUDA importers record a source union; the probe names the Metal ABI, the one the Metal
generator rebuilds a union from, and the layout of a float and a 32-bit integer is the same in every ABI here.

The other languages have no union type: HLSL, Slang, GLSL, GLSL ES and WGSL declare only structs, and Mojo has no
type whose members share storage.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("union",)

_C_BODY = "Bits b;\nb.f = fin[i];\nb.u = (b.u >> 16) | 0x3F800000u;\noutp[i] = b.f;"


def _c_union(unsigned: str) -> str:
    return f"union Bits {{\n    float f;\n    {unsigned} u;\n}};"


KERNELS = {"union": Template(_C_BODY, _c_union("unsigned int"))}
METAL = {"union": Template(_C_BODY, _c_union("uint"))}
OPENCL = {
    "union": Template(_C_BODY, "typedef union {\n    float f;\n    uint u;\n} Bits;")
}
RUST = {
    "union": Template(
        "let mut b = Bits { f: fin[i] };\nunsafe {\n    b.u = (b.u >> 16) | 0x3F80_0000;\n    outp[i] = b.f;\n}",
        "#[derive(Clone, Copy)]\n#[repr(C)]\nunion Bits {\n    f: f32,\n    u: u32,\n}",
    )
}
CROSSGL = {
    "union": Template(
        "Bits b;\nb.f = fin[i];\nb.u = (b.u >> 16u) | 0x3F800000u;\noutp[i] = b.f;",
        prelude=(
            "@union_layout(4, 4, little_endian, metal)\nstruct Bits {\n"
            "    @union_member_layout(0, 4, 4) float f;\n"
            "    @union_member_layout(0, 4, 4) uint u;\n};"
        ),
    )
}

ELSEWHERE = dict.fromkeys(NAMES, ABSENT)
