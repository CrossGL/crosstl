"""User functions named like a CUDA or HIP device library function.

CUDA and HIP output include ``cuda_runtime.h`` or ``hip/hip_runtime.h``, which declare the device math library
(CUDA Math API: single, double and integer functions, such as ``rsqrt``, ``fabs`` or ``sinf``) and the device
runtime functions (atomics, texture and surface fetches, ``printf``, ``make_float3`` and the like) in the global
namespace; HIP mirrors the same API. The generators also write ``transpose`` and ``inverse`` helpers for their
matrix types. A user function with one of those names is a redefinition (nvcc: ``function "rsqrt(float)" has
already been defined``; hipcc: ``redefinition of 'fabs'``) or makes its calls ambiguous. CrossGL calls the user
function, so the CUDA and HIP generators rename it, every overload and every call that one of its overloads accepts
by argument count (issue #3326), as the Metal generator does for the Metal Standard Library
(:mod:`.metal_library_functions`). Identifiers that start with ``__`` are reserved for the implementation in C++
([lex.name]) and are renamed as well. The generators resolve a call by name alone, so a user function is also renamed
when some call of its name has an argument count only a CrossGL built-in of that name takes; that call then reaches
the built-in.
"""

from __future__ import annotations

from copy import deepcopy

from ..semantics.function_name_collisions import (
    names_called_with_other_arity,
    rename_functions,
    user_function_names,
)

__all__ = [
    "CUDA_LIBRARY_FUNCTIONS",
    "rename_cuda_library_functions",
    "shadows_cuda_library",
]

_MATH_FUNCTIONS = (
    "acos acosh asin asinh atan atan2 atanh cbrt ceil copysign cos cosh cospi cyl_bessel_i0 cyl_bessel_i1 erf "
    "erfc erfcinv erfcx erfinv exp exp10 exp2 expm1 fabs fdim floor fma fmax fmin fmod frexp hypot ilogb j0 j1 jn "
    "ldexp lgamma llrint llround log log10 log1p log2 logb lrint lround modf nan nearbyint nextafter norm norm3d "
    "norm4d normcdf normcdfinv pow rcbrt remainder remquo rhypot rint rnorm rnorm3d rnorm4d round rsqrt scalbln "
    "scalbn sin sincos sincospi sinh sinpi sqrt tan tanh tgamma trunc y0 y1 yn"
).split()

# CUDA Math API, "Single Precision Mathematical Functions": the ``f``-suffixed forms, and ``fdividef``.
_SINGLE_PRECISION = frozenset(f"{name}f" for name in _MATH_FUNCTIONS) | {"fdividef"}

# "Double Precision Mathematical Functions", whose C++ overloads also take ``float``; "Integer Mathematical
# Functions"; and the classification functions every precision shares.
_OTHER_MATH = frozenset(_MATH_FUNCTIONS) | {
    "abs",
    "labs",
    "llabs",
    "llmax",
    "llmin",
    "max",
    "min",
    "umax",
    "umin",
    "ullmax",
    "ullmin",
    "isfinite",
    "isinf",
    "isnan",
    "signbit",
}

# "Half Precision Intrinsics" and "Bfloat16 Precision Intrinsics" that do not start with ``__``.
_HALF_PRECISION = frozenset(
    {
        "hceil",
        "hcos",
        "hexp",
        "hexp10",
        "hexp2",
        "hfloor",
        "hlog",
        "hlog10",
        "hlog2",
        "hrcp",
        "hrint",
        "hrsqrt",
        "hsin",
        "hsqrt",
        "htanh",
        "htrunc",
        "h2ceil",
        "h2cos",
        "h2exp",
        "h2exp10",
        "h2exp2",
        "h2floor",
        "h2log",
        "h2log10",
        "h2log2",
        "h2rcp",
        "h2rint",
        "h2rsqrt",
        "h2sin",
        "h2sqrt",
        "h2tanh",
        "h2trunc",
    }
)

# Device runtime functions of the CUDA C++ Programming Guide ("C++ Language Extensions") and the C library
# functions it makes callable from device code.
_DEVICE_RUNTIME = frozenset(
    {
        "assert",
        "clock",
        "clock64",
        "free",
        "malloc",
        "memcpy",
        "memset",
        "printf",
    }
)

# "Atomic Functions", each with its ``_block`` and ``_system`` scope variants.
_ATOMICS = frozenset(
    f"atomic{operation}{scope}"
    for operation in (
        "Add",
        "Sub",
        "Exch",
        "Min",
        "Max",
        "Inc",
        "Dec",
        "CAS",
        "And",
        "Or",
        "Xor",
    )
    for scope in ("", "_block", "_system")
)

# The ``make_<type><N>`` constructors of the built-in vector types ("Built-in Vector Types").
_VECTOR_CONSTRUCTORS = frozenset(
    f"make_{scalar}{lanes}"
    for scalar in (
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "long",
        "ulong",
        "longlong",
        "ulonglong",
        "float",
        "double",
    )
    for lanes in (1, 2, 3, 4)
) | {
    "make_half2",
    "make_bfloat162",
    "make_cudaExtent",
    "make_cudaPitchedPtr",
    "make_cudaPos",
}

# Matrix helpers the CUDA and HIP generators write for their ``floatNxM`` types.
_GENERATOR_HELPERS = frozenset({"inverse", "transpose"})

CUDA_LIBRARY_FUNCTIONS = (
    _SINGLE_PRECISION
    | _OTHER_MATH
    | _HALF_PRECISION
    | _DEVICE_RUNTIME
    | _ATOMICS
    | _VECTOR_CONSTRUCTORS
    | _GENERATOR_HELPERS
)

# Families whose members share a prefix: reserved identifiers (intrinsics such as ``__syncthreads``), and texture
# and surface object functions.
CUDA_LIBRARY_PREFIXES = (
    "__",
    "tex1D",
    "tex2D",
    "tex3D",
    "texCubemap",
    "surf1D",
    "surf2D",
    "surf3D",
    "surfCubemap",
)


def shadows_cuda_library(name) -> bool:
    """Whether a user function called ``name`` would share its name with a CUDA or HIP device library function."""
    return isinstance(name, str) and (
        name in CUDA_LIBRARY_FUNCTIONS or name.startswith(CUDA_LIBRARY_PREFIXES)
    )


def rename_cuda_library_functions(ast):
    """``ast``, or a copy whose user functions named like library functions are renamed with every call to them."""
    names = {name for name in user_function_names(ast) if shadows_cuda_library(name)}
    # The generators treat every call of a declared name as a call of the user function, so one that only a
    # built-in accepts, such as ``clamp(x, 0.0, 1.0)`` beside a user ``clamp(float)``, needs the user one renamed.
    names |= names_called_with_other_arity(ast)
    if not names:
        return ast
    ast = deepcopy(ast)
    rename_functions(ast, names, by_arity=True)
    return ast
