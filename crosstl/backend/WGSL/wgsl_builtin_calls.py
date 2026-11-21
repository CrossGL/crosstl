"""WGSL calls for the WGSL importer: value constructors, conversions, ``bitcast`` and the built-in functions of
W3C WGSL section 17, typed and written as CrossGL.

CrossGL spells most built-in functions as GLSL does (``inverseSqrt`` is ``inversesqrt``, ``countOneBits`` is
``bitCount``), the subgroup and quad operations as HLSL's waves, and textures with separate samplers as the HLSL
importer writes them (``textureSampleLevel(t, s, uv, l)`` is ``textureLod(t, s, uv, l)``). Where WGSL defines a
result CrossGL leaves open, the call is written so that it has WGSL's: ``round`` rounds halfway cases to even,
``clamp`` is ``min(max(e, low), high)``, and ``extractBits``/``insertBits`` clamp their offset and count to the bit
width (section 17.6). A built-in function the importer does not know is refused.
"""

from __future__ import annotations

import re

from .wgsl_types import (
    ABSTRACT_FLOAT,
    ABSTRACT_INT,
    BOOL,
    F16,
    F32,
    I32,
    U32,
    Array,
    Atomic,
    Matrix,
    Pointer,
    ResultStruct,
    Scalar,
    StructType,
    Texture,
    Vector,
    WgslImportError,
    concretize,
    element_scalar,
    is_abstract,
    with_scalar,
)

__all__ = ["call_type", "emit_call"]

# WGSL built-in functions CrossGL spells the same way, whose result has the type of their (unified) arguments.
_SAME_NAME_SAME_TYPE = frozenset(
    """abs acos acosh asin asinh atan atanh ceil cos cosh degrees exp exp2 floor fma fract log log2 max min mix
    pow radians saturate sign sin sinh smoothstep sqrt step tan tanh trunc reflect refract normalize ldexp
    fwidth fwidthCoarse fwidthFine subgroupAdd subgroupAnd subgroupBroadcast subgroupBroadcastFirst
    subgroupExclusiveAdd subgroupExclusiveMul subgroupInclusiveAdd subgroupInclusiveMul subgroupMax subgroupMin
    subgroupMul subgroupOr subgroupShuffle subgroupShuffleDown subgroupShuffleUp subgroupShuffleXor subgroupXor
    """.split()
)
# WGSL built-in functions with another CrossGL name and the type of their (unified) arguments.
_RENAMED_SAME_TYPE = {
    "atan2": "atan",
    "inverseSqrt": "inversesqrt",
    "faceForward": "faceforward",
    "reverseBits": "bitfieldReverse",
    "countLeadingZeros": "clz",
    "countTrailingZeros": "ctz",
    "dpdx": "dFdx",
    "dpdy": "dFdy",
    "dpdxCoarse": "dFdxCoarse",
    "dpdxFine": "dFdxFine",
    "dpdyCoarse": "dFdyCoarse",
    "dpdyFine": "dFdyFine",
    "round": "roundEven",
    "quadBroadcast": "QuadReadLaneAt",
    "quadSwapDiagonal": "QuadReadAcrossDiagonal",
    "quadSwapX": "QuadReadAcrossX",
    "quadSwapY": "QuadReadAcrossY",
}
# GLSL's bit-counting functions return int (vectors) whatever the operand; WGSL returns the operand's type.
_INT_RESULT_SAME_TYPE = {
    "countOneBits": "bitCount",
    "firstLeadingBit": "findMSB",
    "firstTrailingBit": "findLSB",
}
# Functions of fixed result type: (CrossGL name, result type, extra arguments written after the WGSL ones).
_FIXED = {
    "pack4x8snorm": ("packSnorm4x8", U32, ""),
    "pack4x8unorm": ("packUnorm4x8", U32, ""),
    "pack2x16snorm": ("packSnorm2x16", U32, ""),
    "pack2x16unorm": ("packUnorm2x16", U32, ""),
    "pack2x16float": ("packHalf2x16", U32, ""),
    "unpack4x8snorm": ("unpackSnorm4x8", Vector(4, F32), ""),
    "unpack4x8unorm": ("unpackUnorm4x8", Vector(4, F32), ""),
    "unpack2x16snorm": ("unpackSnorm2x16", Vector(2, F32), ""),
    "unpack2x16unorm": ("unpackUnorm2x16", Vector(2, F32), ""),
    "unpack2x16float": ("unpackHalf2x16", Vector(2, F32), ""),
    "pack4xI8": ("pack_s8", U32, ""),
    "pack4xI8Clamp": ("pack_clamp_s8", U32, ""),
    "pack4xU8": ("pack_u8", U32, ""),
    "pack4xU8Clamp": ("pack_u8", U32, ""),
    "unpack4xI8": ("unpack_s8s32", Vector(4, I32), ""),
    "unpack4xU8": ("unpack_u8u32", Vector(4, U32), ""),
    "dot4I8Packed": ("dot4add_i8packed", I32, "0"),
    "dot4U8Packed": ("dot4add_u8packed", U32, "0u"),
    "subgroupBallot": ("subgroupBallot", Vector(4, U32), ""),
    "subgroupElect": ("subgroupElect", BOOL, ""),
    "subgroupAll": ("allInvocations", BOOL, ""),
    "subgroupAny": ("anyInvocation", BOOL, ""),
    "all": ("all", BOOL, ""),
    "any": ("any", BOOL, ""),
}
_BARRIERS = {
    "workgroupBarrier": "barrier",
    "storageBarrier": "memoryBarrierBuffer",
    "textureBarrier": "memoryBarrierImage",
}
_ATOMICS = {
    "atomicAdd": "atomicAdd",
    "atomicSub": "atomicSub",
    "atomicMax": "atomicMax",
    "atomicMin": "atomicMin",
    "atomicAnd": "atomicAnd",
    "atomicOr": "atomicOr",
    "atomicXor": "atomicXor",
    "atomicExchange": "atomicExchange",
}
# Every atomic function and its CrossGL name: CrossGL's atomic functions are GLSL's, with WGSL's names.
_ATOMIC_CALLS = {**_ATOMICS, "atomicLoad": "atomicLoad", "atomicStore": "atomicStore"}
_TEXTURE_FUNCTIONS = frozenset(
    """textureDimensions textureGather textureGatherCompare textureLoad textureNumLayers textureNumLevels
    textureNumSamples textureSample textureSampleBias textureSampleCompare textureSampleCompareLevel
    textureSampleGrad textureSampleLevel textureSampleBaseClampToEdge textureStore""".split()
)
_VECTOR_ALIAS = re.compile(r"vec([234])([iufh])?")
_MATRIX_NAME = re.compile(r"mat([234])x([234])([fh])?")
_SCALAR_TYPES = {"bool": BOOL, "i32": I32, "u32": U32, "f32": F32, "f16": F16}
_SUFFIX = {"i": I32, "u": U32, "f": F32, "h": F16}


def _unified_scalar(types):
    concrete = [
        element_scalar(t)
        for t in types
        if element_scalar(t) is not None and not is_abstract(t)
    ]
    if concrete:
        return concrete[0]
    scalars = [element_scalar(t) for t in types if element_scalar(t) is not None]
    if ABSTRACT_FLOAT in scalars:
        return ABSTRACT_FLOAT
    return ABSTRACT_INT if scalars else None


def _widest(types):
    """The argument type with the most components: the shape of a component-wise result."""
    best = types[0]
    for type_ in types[1:]:
        if isinstance(type_, (Vector, Matrix)) and not isinstance(
            best, (Vector, Matrix)
        ):
            best = type_
    return best


# Typing.


def _constructor_type(converter, callee, arg_types):
    """The type a value constructor builds, or ``None`` when ``callee`` names no type."""
    ctx, name = converter.ctx, callee.name
    if name in ctx.functions:
        return None
    if (
        callee.template_args is not None
        or name in _SCALAR_TYPES
        or name in ctx.structs
        or name in ctx.aliases
    ):
        return ctx.resolve(callee)
    vector = _VECTOR_ALIAS.fullmatch(name)
    if vector:
        size = int(vector.group(1))
        if vector.group(2):
            return Vector(size, _SUFFIX[vector.group(2)])
        return Vector(size, _unified_scalar(arg_types) or ABSTRACT_INT)
    matrix = _MATRIX_NAME.fullmatch(name)
    if matrix:
        scalar = (
            _SUFFIX[matrix.group(3)] if matrix.group(3) else _unified_scalar(arg_types)
        )
        if scalar is None or scalar == ABSTRACT_INT:
            scalar = ABSTRACT_FLOAT if scalar is not None else F32
        return Matrix(int(matrix.group(1)), int(matrix.group(2)), scalar)
    if name == "array":
        if not arg_types:
            raise WgslImportError("array() needs a type or elements", callee.line)
        elem = arg_types[0]
        scalar = _unified_scalar(arg_types)
        if scalar is not None:
            elem = with_scalar(elem, scalar)
        return Array(elem, len(arg_types))
    return None


def _texture_result(name, texture, line):
    if not isinstance(texture, Texture):
        raise WgslImportError(f"{name} needs a texture", line)
    if name == "textureDimensions":
        size = {"1d": 1, "3d": 3}.get(texture.dimension, 2)
        return U32 if size == 1 else Vector(size, U32)
    if name in ("textureNumLayers", "textureNumLevels", "textureNumSamples"):
        return U32
    if name == "textureStore":
        return None
    if name in ("textureSampleCompare", "textureSampleCompareLevel"):
        return F32
    if texture.is_depth and name in (
        "textureLoad",
        "textureSample",
        "textureSampleLevel",
    ):
        return F32
    return Vector(4, texture.sampled)


def _builtin_type(converter, node, arg_types):
    name, line = node.callee.name, node.line
    values = [converter.deref(t) for t in arg_types]
    if (
        name in _SAME_NAME_SAME_TYPE
        or name in _RENAMED_SAME_TYPE
        or name in _INT_RESULT_SAME_TYPE
        or name in ("clamp", "extractBits", "insertBits", "quantizeToF16")
    ):
        if name == "ldexp":
            return values[0]
        if name in (
            "subgroupBroadcast",
            "subgroupShuffle",
            "subgroupShuffleDown",
            "subgroupShuffleUp",
            "subgroupShuffleXor",
            "quadBroadcast",
            "extractBits",
            "insertBits",
        ):
            return values[0]
        return with_scalar(_widest(values), _unified_scalar(values))
    if name in _FIXED:
        return _FIXED[name][1]
    if name in ("length", "distance", "dot", "determinant"):
        return _unified_scalar(values)
    if name == "cross":
        return Vector(3, _unified_scalar(values))
    if name == "transpose":
        return Matrix(values[0].rows, values[0].cols, values[0].elem)
    if name == "select":
        return with_scalar(_widest(values[:2]), _unified_scalar(values[:2]))
    if name in ("frexp", "modf"):
        value = concretize(values[0])
        second = (
            ("exp", with_scalar(value, I32)) if name == "frexp" else ("whole", value)
        )
        return ResultStruct(name, (("fract", value), second))
    if name == "arrayLength":
        return U32
    if name in _ATOMICS or name == "atomicLoad":
        return _atomic_value(values[0], line)
    if name == "atomicStore" or name in _BARRIERS:
        return None
    if name == "atomicCompareExchangeWeak":
        value = _atomic_value(values[0], line)
        return ResultStruct(
            "atomic_compare_exchange", (("old_value", value), ("exchanged", BOOL))
        )
    if name == "workgroupUniformLoad":
        return values[0]
    if name in _TEXTURE_FUNCTIONS:
        index = (
            1 if name == "textureGather" and not isinstance(values[0], Texture) else 0
        )
        return _texture_result(name, values[index], line)
    raise WgslImportError(f"built-in function '{name}' is not supported", line)


def _atomic_value(type_, line):
    if isinstance(type_, Atomic):
        return type_.elem
    raise WgslImportError("an atomic function needs a pointer to an atomic", line)


def call_type(converter, node):
    callee = node.callee
    if callee.name == "bitcast":
        return converter.ctx.resolve(callee.template_args[0])
    arg_types = [converter.type_of(arg) for arg in node.args]
    if callee.name in converter.ctx.functions:
        function = converter.ctx.functions[callee.name]
        return (
            converter.ctx.resolve(function.return_type)
            if function.return_type
            else None
        )
    constructed = _constructor_type(
        converter, callee, [converter.deref(t) for t in arg_types]
    )
    if constructed is not None:
        return constructed
    return _builtin_type(converter, node, arg_types)


# Emission.


def emit_call(converter, node, want):
    callee = node.callee
    if callee.name == "bitcast":
        return _emit_bitcast(converter, node)
    if callee.name in converter.ctx.functions:
        return converter.emit_user_call(node)
    arg_types = [converter.deref(converter.type_of(arg)) for arg in node.args]
    constructed = _constructor_type(converter, callee, arg_types)
    if constructed is not None:
        if is_abstract(constructed) and want is not None:
            constructed = with_scalar(constructed, element_scalar(want))
        return _emit_constructor(converter, node, concretize(constructed))
    return _emit_builtin(converter, node, arg_types, want)


def zero_value(converter, type_, line=None) -> str:
    """The zero value of ``type_`` (section 6.6) as a CrossGL expression."""
    type_ = concretize(type_)
    if isinstance(type_, Scalar):
        return {
            "bool": "false",
            "i32": "0",
            "u32": "0u",
            "f32": "0.0",
            "f16": "half(0.0)",
        }[type_.name]
    if isinstance(type_, Atomic):
        return zero_value(converter, type_.elem, line)
    if isinstance(type_, (Vector, Matrix)):
        return f"{converter.type_text(type_, line)}({zero_value(converter, type_.elem, line)})"
    if isinstance(type_, Array):
        if type_.count is None:
            raise WgslImportError("a runtime-sized array has no zero value", line)
        element = zero_value(converter, type_.elem, line)
        return f"{converter.type_expression(type_, line)}({', '.join([element] * type_.count)})"
    if isinstance(type_, StructType):
        fields = converter.ctx.struct_fields(type_.name)
        values = ", ".join(
            zero_value(converter, field_type, line) for _n, field_type, _a, _l in fields
        )
        return f"{converter.type_text(type_, line)}({values})"
    raise WgslImportError(f"{type_} has no zero value", line)


def _emit_constructor(converter, node, type_):
    line = node.line
    if not node.args:
        return zero_value(converter, type_, line)
    if isinstance(type_, Scalar):
        arg = node.args[0]
        if is_abstract(converter.type_of(arg)):
            return converter.emit(arg, type_)
        return f"{converter.type_text(type_, line)}({converter.emit(arg)})"
    scalar = element_scalar(type_)
    if isinstance(type_, Array):
        args = ", ".join(converter.emit_value(arg, type_.elem) for arg in node.args)
        return f"{converter.type_expression(type_, line)}({args})"
    if isinstance(type_, StructType):
        fields = converter.ctx.struct_fields(type_.name)
        args = ", ".join(
            converter.emit_value(arg, field[1]) for arg, field in zip(node.args, fields)
        )
        return f"{converter.type_text(type_, line)}({args})"
    args = []
    for arg in node.args:
        arg_type = converter.deref(converter.type_of(arg))
        args.append(
            converter.emit(
                arg, with_scalar(arg_type, scalar) if is_abstract(arg_type) else None
            )
        )
    if (
        isinstance(type_, Vector)
        and len(node.args) == 1
        and isinstance(converter.deref(converter.type_of(node.args[0])), Scalar)
    ):
        return f"{converter.type_text(type_, line)}({args[0]})"
    if isinstance(type_, Matrix) and len(node.args) == 1:
        # A matrix from a matrix converts it (section 17.1.2); CrossGL's mat(m) does too.
        return f"{converter.type_text(type_, line)}({args[0]})"
    return f"{converter.type_text(type_, line)}({', '.join(args)})"


_BITCASTS = {
    ("f32", "u32"): "floatBitsToUint",
    ("f32", "i32"): "floatBitsToInt",
    ("u32", "f32"): "uintBitsToFloat",
    ("i32", "f32"): "intBitsToFloat",
    ("u32", "i32"): None,
    ("i32", "u32"): None,
}


def _emit_bitcast(converter, node):
    target = converter.ctx.resolve(node.callee.template_args[0])
    source = concretize(converter.deref(converter.type_of(node.args[0])))
    value = converter.emit(node.args[0])
    if _shape_of(source) != _shape_of(target):
        raise WgslImportError(
            "bitcast between types of different shapes is not supported", node.line
        )
    key = (element_scalar(source).name, element_scalar(target).name)
    if key[0] == key[1]:
        return value
    if key not in _BITCASTS:
        raise WgslImportError(
            f"bitcast from {key[0]} to {key[1]} is not supported", node.line
        )
    function = _BITCASTS[key]
    if function is None:
        # i32 and u32 conversions keep the bits (two's complement), as bitcast does.
        return f"{converter.type_text(target, node.line)}({value})"
    return f"{function}({value})"


def _shape_of(type_):
    if isinstance(type_, Vector):
        return ("vector", type_.size)
    if isinstance(type_, Scalar):
        return ("scalar",)
    return (type(type_).__name__,)


def _emit_args(converter, node, arg_types, scalar=None):
    """The arguments, an abstract one converted to ``scalar`` (by default, the unified scalar of them all)."""
    if scalar is None:
        scalar = (
            concretize(_unified_scalar(arg_types))
            if _unified_scalar(arg_types)
            else None
        )
    emitted = []
    for arg, arg_type in zip(node.args, arg_types):
        if is_abstract(arg_type) and scalar is not None:
            emitted.append(converter.emit(arg, with_scalar(arg_type, scalar)))
        else:
            emitted.append(converter.emit(arg))
    return emitted


def _emit_builtin(converter, node, arg_types, want):
    name, line = node.callee.name, node.line
    result = concretize(
        _builtin_type(converter, node, [converter.type_of(a) for a in node.args])
    )
    want_scalar = element_scalar(want) if want is not None else None
    scalar = (
        element_scalar(result)
        if element_scalar(result) and not is_abstract(result)
        else want_scalar
    )
    if name in _SAME_NAME_SAME_TYPE or name in _RENAMED_SAME_TYPE:
        args = _emit_args(
            converter, node, arg_types, _arg_scalar(name, arg_types, scalar)
        )
        if name == "ldexp":
            args = _emit_args(converter, node, arg_types[:1], scalar)[:1] + [
                converter.emit(
                    node.args[1],
                    (
                        with_scalar(arg_types[1], I32)
                        if is_abstract(arg_types[1])
                        else None
                    ),
                )
            ]
        crossgl_name = _RENAMED_SAME_TYPE[name] if name in _RENAMED_SAME_TYPE else name
        return f"{crossgl_name}({', '.join(args)})"
    if name in _INT_RESULT_SAME_TYPE:
        args = _emit_args(converter, node, arg_types, scalar)
        call = f"{_INT_RESULT_SAME_TYPE[name]}({', '.join(args)})"
        return (
            f"{converter.type_text(result, line)}({call})"
            if element_scalar(result) == U32
            else call
        )
    if name == "clamp":
        e, low, high = _emit_args(converter, node, arg_types, scalar)
        return f"min(max({e}, {low}), {high})"
    if name in ("extractBits", "insertBits"):
        return _emit_bit_field(converter, node, arg_types, scalar)
    if name == "quantizeToF16":
        (arg,) = _emit_args(converter, node, arg_types, F32)
        return f"f16tof32(f32tof16({arg}))"
    if name == "pack4xU8Clamp":
        (arg,) = _emit_args(converter, node, arg_types, U32)
        return f"pack_u8(min({arg}, uvec4(255u)))"
    if name in _FIXED:
        crossgl, _type, extra = _FIXED[name]
        args = _emit_args(converter, node, arg_types)
        return f"{crossgl}({', '.join(args + ([extra] if extra else []))})"
    if name in ("length", "distance", "dot", "determinant", "cross", "transpose"):
        return f"{name}({', '.join(_emit_args(converter, node, arg_types, scalar))})"
    if name == "select":
        return _emit_select(converter, node, arg_types, scalar)
    return _emit_memory_builtin(converter, node, arg_types, result)


def _arg_scalar(name, arg_types, scalar):
    if name in (
        "subgroupBroadcast",
        "subgroupShuffle",
        "subgroupShuffleDown",
        "subgroupShuffleUp",
        "subgroupShuffleXor",
        "quadBroadcast",
    ):
        return None
    return scalar


def _emit_bit_field(converter, node, arg_types, scalar):
    name = node.callee.name
    count = 2 if name == "extractBits" else 3
    values = _emit_args(converter, node, arg_types[: count - 1], scalar)
    offset_node, bits_node = node.args[count - 1], node.args[count]
    offset_value, bits_value = converter.ctx.eval_const(
        offset_node
    ), converter.ctx.eval_const(bits_node)
    if isinstance(offset_value, int) and isinstance(bits_value, int):
        offset = min(offset_value, 32)
        bits = min(bits_value, 32 - offset)
        offset_text, bits_text = str(offset), str(bits)
    else:
        offset = converter.emit(offset_node, U32)
        bits = converter.emit(bits_node, U32)
        if not converter.is_stable(offset_node):
            offset = converter.hoist(U32, offset, node.line)
        offset_text = f"int(min({offset}, 32u))"
        bits_text = f"int(min({bits}, 32u - min({offset}, 32u)))"
    crossgl = "bitfieldExtract" if name == "extractBits" else "bitfieldInsert"
    return f"{crossgl}({', '.join(values)}, {offset_text}, {bits_text})"


def _emit_select(converter, node, arg_types, scalar):
    false_value, true_value = _emit_args(converter, node, arg_types[:2], scalar)
    condition_type = arg_types[2]
    condition = converter.emit(node.args[2])
    if isinstance(condition_type, Vector):
        return f"mix({false_value}, {true_value}, {condition})"
    if converter.has_call(node.args[0]) or converter.has_call(node.args[1]):
        # select evaluates both operands; a conditional expression evaluates one.
        false_value = converter.hoist(concretize(arg_types[0]), false_value, node.line)
        true_value = converter.hoist(concretize(arg_types[1]), true_value, node.line)
    return f"({condition} ? {true_value} : {false_value})"


def _emit_memory_builtin(converter, node, arg_types, result):
    name, line = node.callee.name, node.line
    if name in _BARRIERS:
        return f"{_BARRIERS[name]}()"
    if name == "arrayLength":
        return f"buffer_dimensions({converter.emit(node.args[0])})"
    if name in _ATOMICS or name in ("atomicLoad", "atomicStore"):
        target = converter.emit(node.args[0])
        values = [
            converter.emit_value(
                arg,
                (
                    result
                    if result is not None
                    else _atomic_value(
                        (
                            arg_types[0].elem
                            if isinstance(arg_types[0], Pointer)
                            else arg_types[0]
                        ),
                        line,
                    )
                ),
            )
            for arg in node.args[1:]
        ]
        return f"{_ATOMIC_CALLS[name]}({', '.join([target, *values])})"
    if name == "workgroupUniformLoad":
        return converter.hoist_uniform_load(node, result)
    if name in _TEXTURE_FUNCTIONS:
        from .wgsl_texture_calls import emit_texture_call

        return emit_texture_call(converter, node, arg_types)
    if name in ("frexp", "modf", "atomicCompareExchangeWeak"):
        return converter.result_struct_call(node, result)
    raise WgslImportError(f"built-in function '{name}' is not supported", line)
