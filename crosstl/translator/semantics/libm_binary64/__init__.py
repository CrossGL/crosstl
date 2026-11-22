"""Binary64 math functions for targets whose built-ins take binary32 operands at most.

GLSL, SPIR-V (GLSL.std.450) and HLSL compute ``double`` arithmetic, but declare their transcendental functions for
16- and 32-bit operands only, so a CrossGL program that calls ``exp``, ``log`` or ``pow`` on ``double`` values (as
CUDA, HIP and OpenCL C programs do) needs its own binary64 implementation there. This package holds those
implementations, written once in CrossGL binary64 arithmetic, one module per function group:

* :mod:`.bits`: word access, exact constants and the final scaling;
* :mod:`.exp_log`: ``exp`` and ``log`` (fdlibm e_exp.c and e_log.c);
* :mod:`.power`: ``pow``, ``exp2`` and ``log2`` (fdlibm e_pow.c);
* :mod:`.lowering`: the pass that rewrites binary64 calls on the OpenGL, Vulkan and DirectX targets.
"""

from .lowering import (
    BINARY64_TARGETS,
    FUNCTIONS,
    Binary64MathError,
    helper_program_source,
    lower_binary64_math,
)

__all__ = [
    "BINARY64_TARGETS",
    "Binary64MathError",
    "FUNCTIONS",
    "helper_program_source",
    "lower_binary64_math",
]
