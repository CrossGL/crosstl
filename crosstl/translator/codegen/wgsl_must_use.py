"""Function call statements of WGSL ``@must_use`` built-ins.

WGSL declares most built-in functions that return a value ``@must_use`` (WGSL, "must_use"), so a function call
statement that calls one, such as ``min(a, 4u);``, is rejected. Source languages allow the call as a statement and
throw the value away. :func:`wgsl_call_statement` keeps such a call and discards its value with a phony assignment,
``_ = min(a, 4u);``, which WGSL provides for exactly this case. The arguments are still evaluated, so any side effect
in them is kept.

Atomics, barriers, ``textureStore`` and functions the shader declares itself are written unchanged.
"""

from __future__ import annotations

__all__ = ["WGSL_MUST_USE_BUILTINS", "called_function_name", "wgsl_call_statement"]

# Built-in functions declared ``@must_use`` in the WGSL specification, value constructors included.
WGSL_MUST_USE_BUILTINS = frozenset(
    {
        "abs",
        "acos",
        "acosh",
        "all",
        "any",
        "array",
        "arrayLength",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "bitcast",
        "bool",
        "ceil",
        "clamp",
        "cos",
        "cosh",
        "countLeadingZeros",
        "countOneBits",
        "countTrailingZeros",
        "cross",
        "degrees",
        "determinant",
        "distance",
        "dot",
        "dot4I8Packed",
        "dot4U8Packed",
        "dpdx",
        "dpdxCoarse",
        "dpdxFine",
        "dpdy",
        "dpdyCoarse",
        "dpdyFine",
        "exp",
        "exp2",
        "extractBits",
        "f16",
        "f32",
        "faceForward",
        "firstLeadingBit",
        "firstTrailingBit",
        "floor",
        "fma",
        "fract",
        "frexp",
        "fwidth",
        "fwidthCoarse",
        "fwidthFine",
        "i32",
        "insertBits",
        "inverseSqrt",
        "ldexp",
        "length",
        "log",
        "log2",
        "mat2x2",
        "mat2x3",
        "mat2x4",
        "mat3x2",
        "mat3x3",
        "mat3x4",
        "mat4x2",
        "mat4x3",
        "mat4x4",
        "max",
        "min",
        "mix",
        "modf",
        "normalize",
        "pack2x16float",
        "pack2x16snorm",
        "pack2x16unorm",
        "pack4x8snorm",
        "pack4x8unorm",
        "pack4xI8",
        "pack4xI8Clamp",
        "pack4xU8",
        "pack4xU8Clamp",
        "pow",
        "quadBroadcast",
        "quadSwapDiagonal",
        "quadSwapX",
        "quadSwapY",
        "quantizeToF16",
        "radians",
        "reflect",
        "refract",
        "reverseBits",
        "round",
        "saturate",
        "select",
        "sign",
        "sin",
        "sinh",
        "smoothstep",
        "sqrt",
        "step",
        "subgroupAdd",
        "subgroupAll",
        "subgroupAnd",
        "subgroupAny",
        "subgroupBallot",
        "subgroupBroadcast",
        "subgroupBroadcastFirst",
        "subgroupElect",
        "subgroupExclusiveAdd",
        "subgroupExclusiveMul",
        "subgroupInclusiveAdd",
        "subgroupInclusiveMul",
        "subgroupMax",
        "subgroupMin",
        "subgroupMul",
        "subgroupOr",
        "subgroupShuffle",
        "subgroupShuffleDown",
        "subgroupShuffleUp",
        "subgroupShuffleXor",
        "subgroupXor",
        "tan",
        "tanh",
        "textureDimensions",
        "textureGather",
        "textureGatherCompare",
        "textureLoad",
        "textureNumLayers",
        "textureNumLevels",
        "textureNumSamples",
        "textureSample",
        "textureSampleBaseClampToEdge",
        "textureSampleBias",
        "textureSampleCompare",
        "textureSampleCompareLevel",
        "textureSampleGrad",
        "textureSampleLevel",
        "transpose",
        "trunc",
        "u32",
        "unpack2x16float",
        "unpack2x16snorm",
        "unpack2x16unorm",
        "unpack4x8snorm",
        "unpack4x8unorm",
        "unpack4xI8",
        "unpack4xU8",
        "vec2",
        "vec3",
        "vec4",
        "workgroupUniformLoad",
    }
)


def called_function_name(expression: str) -> str | None:
    """The callee of ``expression`` when the whole expression is one call, such as ``vec2<f32>(a, b)``."""
    text = expression.strip()
    if not text.endswith(")"):
        return None
    index = 0
    while index < len(text) and (text[index].isalnum() or text[index] == "_"):
        index += 1
    name = text[:index]
    if not name or name[0].isdigit():
        return None
    if index < len(text) and text[index] == "<":
        depth = 0
        while index < len(text):
            if text[index] == "<":
                depth += 1
            elif text[index] == ">":
                depth -= 1
                if depth == 0:
                    index += 1
                    break
            index += 1
    if index >= len(text) or text[index] != "(":
        return None
    depth = 0
    for position in range(index, len(text)):
        if text[position] == "(":
            depth += 1
        elif text[position] == ")":
            depth -= 1
            if depth == 0:
                # The parenthesis that opens the argument list closes at the end of the expression.
                return name if position == len(text) - 1 else None
    return None


def wgsl_call_statement(expression: str, user_functions=()) -> str:
    """``expression`` as a function call statement, with a phony assignment for ``@must_use`` built-ins.

    ``user_functions`` names the functions the shader declares. One of them may shadow a built-in, such as a
    ``void sign(uint i)``, and a call to it is written unchanged.
    """
    name = called_function_name(expression)
    if name in WGSL_MUST_USE_BUILTINS and name not in user_functions:
        return f"_ = {expression}"
    return expression
