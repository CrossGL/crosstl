"""Device-global probe: a program-scope variable in device memory with an initializer, which every invocation of
every launch shares.

OpenCL C declares it ``__global`` (OpenCL C 3.0, "Program Scope Global Variables", the
``__opencl_c_program_scope_global_variables`` feature), and CUDA and HIP ``__device__``. The probe initializes the
first three elements of an array, and each invocation reads and rewrites only its own element, so the result does
not depend on the order in which invocations run. CrossGL's probe qualifies the variable with the ``device`` address
space.

The other languages have no such variable. Metal declares program-scope variables only in the constant address
space. An unqualified global is a uniform in HLSL and Slang, and with ``static`` one copy per invocation, as a private
global is in GLSL, GLSL ES and WGSL. WGSL storage and GLSL buffer blocks take no initializer, and rust-gpu and Mojo
kernels reach device memory only through the buffers they are passed.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("device-global",)

_BODY = "acc[i & 63{u}] = acc[i & 63{u}] * 2.0{f} + fin[i];\noutp[i] = acc[i & 63{u}];"
_DECLARATION = "{space} float acc[64] = {{1.0{f}, 2.0{f}, 4.0{f}}};"


def _c(space: str, f: str = "f", u: str = "") -> dict[str, Template]:
    return {
        "device-global": Template(
            _BODY.format(f=f, u=u), _DECLARATION.format(space=space, f=f)
        )
    }


KERNELS = _c("__device__")
OPENCL = _c("__global")
CROSSGL = _c("device", f="", u="u")

ELSEWHERE = dict.fromkeys(NAMES, ABSENT)
