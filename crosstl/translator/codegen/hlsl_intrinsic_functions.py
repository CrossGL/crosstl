"""User functions named like an HLSL intrinsic function.

DXC resolves a call against the intrinsic's overloads and the program's own functions together (DXC's intrinsic
table, ``utils/hct/gen_intrin_main.txt``, declares them as free functions of the global scope). Where a program
function takes what an intrinsic overload takes, the intrinsic wins: beside a helper ``float frac(float x)`` the call
``frac(x)`` reads the fractional part, and the helper never runs. The output compiles and computes something else.
The helper also captures a built-in the generator writes under the same name, such as ``frac`` for CrossGL's
``fract``.

CrossGL calls the user function, so the HLSL generator renames it, every overload and every call that one of its
overloads accepts by argument count and vector shape, as the Metal and CUDA generators do for their libraries
(:mod:`.metal_library_functions`, :mod:`.cuda_library_functions`). A call with another argument count, such as
``max(a, b)`` beside a user ``float max(float x)``, and the intrinsic calls the generator writes itself, keep the
intrinsic. The names are DXC's global intrinsics
(:data:`~crosstl.translator.semantics.builtin_function_names.HLSL_INTRINSIC_FUNCTION_NAMES`).
"""

from __future__ import annotations

from copy import deepcopy

from ..semantics.builtin_function_names import HLSL_INTRINSIC_FUNCTION_NAMES
from ..semantics.function_name_collisions import rename_functions, user_function_names
from ..semantics.hub_types import iter_functions

__all__ = ["rename_hlsl_intrinsic_functions", "shadows_hlsl_intrinsic"]


def shadows_hlsl_intrinsic(name) -> bool:
    """Whether a user function called ``name`` would share its name with a global HLSL intrinsic."""
    return isinstance(name, str) and name in HLSL_INTRINSIC_FUNCTION_NAMES


def _generic_function_names(ast):
    """Names with a generic definition: the generator writes each instantiation under a name of its own."""
    return {
        function.name
        for function in iter_functions(ast)
        if getattr(function, "generic_params", None)
    }


def rename_hlsl_intrinsic_functions(ast):
    """``ast``, or a copy whose user functions named like HLSL intrinsics are renamed with the calls to them."""
    names = {name for name in user_function_names(ast) if shadows_hlsl_intrinsic(name)}
    # A generic primary such as ``generic<T> T clip(T x)`` is reached by its explicit specializations and by
    # ``clip<vec4>(x)`` calls, which keep the primary's name; its instantiations are already named apart.
    names -= _generic_function_names(ast)
    if not names:
        return ast
    ast = deepcopy(ast)
    rename_functions(ast, names, by_arity=True)
    return ast
