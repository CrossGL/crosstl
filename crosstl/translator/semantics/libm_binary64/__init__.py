"""Binary64 math functions for targets whose built-ins take binary32 operands at most.

GLSL, SPIR-V (GLSL.std.450) and HLSL compute ``double`` arithmetic, but declare their transcendental functions for
16- and 32-bit operands only, so a CrossGL program that calls ``exp``, ``log``, ``pow``, ``asin``, ``sinh`` or ``round`` on
``double`` values (as CUDA, HIP and OpenCL C programs do) needs its own binary64 implementation there. This package
holds those implementations, written once in CrossGL binary64 arithmetic, one module per function group:

* :mod:`.bits`: word access, exact constants, ``fabs`` and the final scaling;
* :mod:`.exp_log`: ``exp`` and ``log`` (fdlibm e_exp.c and e_log.c);
* :mod:`.power`: ``pow``, ``exp2`` and ``log2`` (fdlibm e_pow.c);
* :mod:`.roots`: a correctly rounded square root for the helpers that take one;
* :mod:`.inverse_trig`: ``asin``, ``acos``, ``atan`` and ``atan2`` (fdlibm e_asin.c, e_acos.c, s_atan.c, e_atan2.c);
* :mod:`.extended`: ``e^x`` as a double-double, for the hyperbolic functions that round once;
* :mod:`.hyperbolic`: ``sinh``, ``cosh`` and ``tanh`` from Taylor series and :mod:`.extended`, and ``asinh``,
  ``acosh`` and ``atanh`` with ``expm1`` and ``log1p`` (fdlibm s_asinh.c, e_acosh.c, e_atanh.c, s_expm1.c, s_log1p.c);
* :mod:`.rounding`: ``round`` (halfway cases away from zero) in word arithmetic;
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
