"""Pointer-local and pointer-arithmetic probes: a local variable that holds the address of a buffer element, and
pointers moved by adding and subtracting offsets.

The pointer-local probe takes the address of an output element, writes it through the pointer and reads it back.
CUDA, HIP, OpenCL C, Metal and Slang spell it with ``&`` and ``*``, OpenCL C and Metal naming the pointer's address
space, WGSL with a ``let`` bound to ``&outp[i]``, Rust with a mutable reference, and Mojo with an ``UnsafePointer``
to the element. The pointer-arithmetic probe adds the invocation's index to a buffer pointer, subtracts it again,
indexes and dereferences the results, and assigns a moved pointer to a pointer variable. slangc has no compound
assignment for pointers, so every probe assigns ``q = q + i``.

GLSL, GLSL ES and HLSL have no pointer types. WGSL has pointers but no pointer arithmetic, and so does rust-gpu, whose
buffers are slices and which compiles no raw-pointer offsets in Vulkan's logical addressing.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("pointer-local", "pointer-arithmetic")

# Both probes leave 10 * fin[i] + 1 and 4 * fin[i] + 1 in the output.
_LOCAL = (
    "{space}float* p = &outp[i];\n*p = fin[i] * 2.0{f};\n*p += *p * 4.0{f} + 1.0{f};"
)
_ARITHMETIC = (
    "{space}float* p = {base} + i;\n*p = fin[i];\n{space}float* q = p - i;\nq[i] = *(q + i) * 4.0{f};\n"
    "q = q + i;\n*q += 1.0{f};"
)


def _c(space: str, base: str = "outp", f: str = "f") -> dict[str, Template]:
    return {
        "pointer-local": Template(_LOCAL.format(space=space, f=f)),
        "pointer-arithmetic": Template(_ARITHMETIC.format(space=space, base=base, f=f)),
    }


KERNELS = _c("")
OPENCL = _c("__global ")
METAL = _c("device ")
# A structured buffer is no pointer, so Slang's arithmetic starts from its first element's address.
SLANG = _c("", base="&outp[0]")
CROSSGL = _c("", f="")

WGSL = {
    "pointer-local": Template(
        "let p = &outp[i];\n*p = fin[i] * 2.0;\n*p += *p * 4.0 + 1.0;"
    ),
    "pointer-arithmetic": ABSENT,
}

RUST = {
    "pointer-local": Template(
        "let p = &mut outp[i];\n*p = fin[i] * 2.0;\n*p += *p * 4.0 + 1.0;"
    ),
    "pointer-arithmetic": ABSENT,
}

MOJO = {
    "pointer-local": Template(
        "var p = UnsafePointer(to=outp[i])\np[] = fin[i] * 2.0\np[] += p[] * 4.0 + 1.0"
    ),
    "pointer-arithmetic": Template(
        "var p = outp + i\np[] = fin[i]\nvar q = p - i\nq[i] = (q + i)[] * 4.0\nq = q + i\nq[] += 1.0"
    ),
}

ELSEWHERE = dict.fromkeys(NAMES, ABSENT)
