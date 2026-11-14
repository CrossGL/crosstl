"""Calls that only a built-in accepts, beside a user function of the built-in's name, in SPIR-V output.

A program may overload a built-in with a function of its own: llama.cpp's Metal sources declare
``static inline float dot(float x, float y)`` beside the library's ``dot(floatN, floatN)`` and call both. A call that
no user overload accepts, such as ``dot(v, w)`` on ``float4`` values, resolves to the built-in, because no implicit
conversion turns a vector into a scalar (see :mod:`~crosstl.translator.semantics.builtin_overload_shapes`).

The SPIR-V generator calls a user function by name whenever one is declared, so such a call became an
``OpFunctionCall`` whose arguments the callee's parameters do not take, which spirv-val rejects. Before generation,
:func:`separate_builtin_overloads` renames each user function with a call that only the built-in accepts, every
overload of it and every call one of them takes; the built-in-only calls keep the name and lower to the built-in.
"""

from __future__ import annotations

from .. import ast as A
from ..semantics.builtin_overload_shapes import calls_only_builtins_accept
from ..semantics.function_name_collisions import rename_functions, user_function_names
from ..semantics.hub_types import call_name

__all__ = ["VALUE_BUILTIN_FUNCTIONS", "separate_builtin_overloads"]

# CrossGL spells its built-in functions as GLSL does. These are the functions of GLSL 4.60 sections 8.1 to 8.8
# (angle and trigonometry, exponential, common, floating-point pack and unpack, geometric, matrix, vector
# relational and integer functions), whose overloads differ by vector width, with the HLSL spellings CrossGL also
# accepts. A user function of another name has no built-in to fall back on, so its calls always reach it.
VALUE_BUILTIN_FUNCTIONS = frozenset(
    {
        # 8.1 angle and trigonometry
        "radians", "degrees", "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        # 8.2 exponential
        "pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
        # 8.3 common
        "abs", "sign", "floor", "trunc", "round", "roundEven", "ceil", "fract", "mod",
        "modf", "min", "max", "clamp", "mix", "step", "smoothstep", "isnan", "isinf",
        "floatBitsToInt", "floatBitsToUint", "intBitsToFloat", "uintBitsToFloat",
        "fma", "frexp", "ldexp",
        # 8.4 floating-point pack and unpack
        "packUnorm2x16", "packSnorm2x16", "packUnorm4x8", "packSnorm4x8",
        "unpackUnorm2x16", "unpackSnorm2x16", "unpackUnorm4x8", "unpackSnorm4x8",
        "packHalf2x16", "unpackHalf2x16", "packDouble2x32", "unpackDouble2x32",
        # 8.5 geometric
        "length", "distance", "dot", "cross", "normalize", "faceforward", "reflect", "refract",
        # 8.6 matrix
        "matrixCompMult", "outerProduct", "transpose", "determinant", "inverse",
        # 8.7 vector relational
        "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual", "equal",
        "notEqual", "any", "all", "not",
        # 8.8 integer
        "uaddCarry", "usubBorrow", "umulExtended", "imulExtended", "bitfieldExtract",
        "bitfieldInsert", "bitfieldReverse", "bitCount", "findLSB", "findMSB",
        # HLSL spellings
        "atan2", "frac", "lerp", "rsqrt", "saturate", "mad", "fmod",
    }
)  # fmt: skip


def separate_builtin_overloads(program) -> dict:
    """Rename, in place, the user functions named like a built-in that has a call of its own; returns old -> new."""
    if not isinstance(program, A.ShaderNode):
        return {}
    builtin_calls = calls_only_builtins_accept(
        program, user_function_names(program) & VALUE_BUILTIN_FUNCTIONS
    )
    if not builtin_calls:
        return {}
    names = {
        call_name(node)
        for node in program.walk()
        if isinstance(node, A.FunctionCallNode) and id(node) in builtin_calls
    }
    return rename_functions(program, names, by_arity=True)
