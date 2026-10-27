"""User functions named like a Metal Standard Library function.

Metal output writes ``using namespace metal;``, which brings the Metal Standard Library (MSL 3.1, chapter 6) into
the program's scope. A user function named ``fwidth`` or ``clamp`` then makes a call ambiguous, as in
``call to 'fwidth' is ambiguous``, or lets the library overload win silently. CrossGL calls the user function, so
the Metal generator renames it, every overload and every call that one of its overloads accepts by argument count
(issue #3156), the way :func:`~crosstl.translator.semantics.function_name_collisions.rename_colliding_functions`
renames functions that share a name with a value. A call with another argument count, such as ``clamp(x, 0.0, 1.0)``
beside a user ``clamp(float)``, and the library calls the generator writes itself, reach the library.
"""

from __future__ import annotations

from copy import deepcopy

from ..semantics.function_name_collisions import rename_functions, user_function_names

__all__ = [
    "MSL_LIBRARY_FUNCTIONS",
    "rename_msl_library_functions",
    "shadows_msl_library",
]

# MSL 3.1 sections 6.2 to 6.9 and 6.13: the free functions of the math, common, integer, relational, geometric,
# derivative, matrix and synchronization libraries.
MSL_LIBRARY_FUNCTIONS = frozenset(
    {
        # 6.5 math
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "ceil",
        "copysign",
        "cos",
        "cosh",
        "cospi",
        "divide",
        "exp",
        "exp10",
        "exp2",
        "fabs",
        "fdim",
        "floor",
        "fma",
        "fmax",
        "fmax3",
        "fmedian3",
        "fmin",
        "fmin3",
        "fmod",
        "fract",
        "frexp",
        "ilogb",
        "ldexp",
        "log",
        "log10",
        "log2",
        "modf",
        "nextafter",
        "pow",
        "powr",
        "rint",
        "round",
        "rsqrt",
        "sin",
        "sincos",
        "sinh",
        "sinpi",
        "sqrt",
        "tan",
        "tanh",
        "tanpi",
        "trunc",
        # 6.3 common
        "clamp",
        "mix",
        "saturate",
        "sign",
        "smoothstep",
        "step",
        # 6.4 integer
        "abs",
        "absdiff",
        "addsat",
        "clz",
        "ctz",
        "extract_bits",
        "hadd",
        "insert_bits",
        "mad24",
        "madhi",
        "madsat",
        "max",
        "max3",
        "median3",
        "min",
        "min3",
        "mul24",
        "mulhi",
        "popcount",
        "reverse_bits",
        "rhadd",
        "rotate",
        "subsat",
        # 6.6 relational
        "all",
        "any",
        "isfinite",
        "isinf",
        "isnan",
        "isnormal",
        "isordered",
        "isunordered",
        "not",
        "select",
        "signbit",
        # 6.7 geometric
        "cross",
        "distance",
        "distance_squared",
        "dot",
        "faceforward",
        "length",
        "length_squared",
        "normalize",
        "reflect",
        "refract",
        # 6.9 derivatives
        "dfdx",
        "dfdy",
        "fwidth",
        # 6.8 matrix
        "determinant",
        "transpose",
        # 6.13 synchronization, and the reinterpreting cast
        "threadgroup_barrier",
        "simdgroup_barrier",
        "as_type",
    }
)

# Families whose members share a prefix: SIMD-group and quad-group functions, atomics, and pack/unpack.
MSL_LIBRARY_PREFIXES = ("simd_", "quad_", "atomic_", "pack_", "unpack_")


def shadows_msl_library(name) -> bool:
    """Whether a user function called ``name`` would share its name with a Metal Standard Library function."""
    return isinstance(name, str) and (
        name in MSL_LIBRARY_FUNCTIONS or name.startswith(MSL_LIBRARY_PREFIXES)
    )


def rename_msl_library_functions(ast):
    """``ast``, or a copy whose user functions named like library functions are renamed with every call to them."""
    names = {name for name in user_function_names(ast) if shadows_msl_library(name)}
    if not names:
        return ast
    ast = deepcopy(ast)
    rename_functions(ast, names, by_arity=True)
    return ast
