"""CrossGL ``as_type<T>(value)`` in CUDA and HIP output.

``as_type<T>(v)`` reinterprets the bits of ``v`` as a value of ``T``, which must have the same size (MSL 4.1 section
2.19 "Type Conversions and Reinterpreting Data"; the Metal importer and the directed-rounding functions of the CUDA
and HIP importers write it). The CUDA and HIP generators used to emit the call verbatim, which neither compiler
declares. A scalar reinterpretation lowers to the bit-cast intrinsics both runtimes declare (CUDA Math API "Type
Casting Intrinsics"; ``hip/amd_detail/amd_device_functions.h`` and ``amd_hip_fp16.h``): ``__float_as_uint``,
``__double_as_longlong``, ``__half_as_ushort`` and their inverses, with a ``static_cast`` between the signed and
unsigned integers of one width. A vector of 32-bit components lowers through the generator's per-component helper.
Any other operand, such as one of a different size or a matrix, is refused.
"""

from __future__ import annotations

import re

__all__ = ["as_type_result_type", "is_as_type_call", "lower_as_type"]

_AS_TYPE = re.compile(r"^as_type<\s*([^<>]+?)\s*>$")

# C spelling -> (size in bits, kind)
_SCALARS = {
    "float": (32, "float"),
    "int": (32, "int"),
    "unsigned int": (32, "uint"),
    "double": (64, "float"),
    "long long": (64, "int"),
    "unsigned long long": (64, "uint"),
    "half": (16, "float"),
    "__half": (16, "float"),
    "short": (16, "int"),
    "unsigned short": (16, "uint"),
}

_INTRINSICS = {
    32: {
        ("float", "int"): "__float_as_int({})",
        ("float", "uint"): "__float_as_uint({})",
        ("int", "float"): "__int_as_float({})",
        ("uint", "float"): "__uint_as_float({})",
        ("int", "uint"): "static_cast<unsigned int>({})",
        ("uint", "int"): "static_cast<int>({})",
    },
    64: {
        ("float", "int"): "__double_as_longlong({})",
        ("float", "uint"): "static_cast<unsigned long long>(__double_as_longlong({}))",
        ("int", "float"): "__longlong_as_double({})",
        ("uint", "float"): "__longlong_as_double(static_cast<long long>({}))",
        ("int", "uint"): "static_cast<unsigned long long>({})",
        ("uint", "int"): "static_cast<long long>({})",
    },
    16: {
        ("float", "int"): "__half_as_short({})",
        ("float", "uint"): "__half_as_ushort({})",
        ("int", "float"): "__short_as_half({})",
        ("uint", "float"): "__ushort_as_half({})",
        ("int", "uint"): "static_cast<unsigned short>({})",
        ("uint", "int"): "static_cast<short>({})",
    },
}


def is_as_type_call(func_name):
    return isinstance(func_name, str) and _AS_TYPE.match(func_name) is not None


def _target_type(func_name):
    return _AS_TYPE.match(func_name).group(1)


def as_type_result_type(generator, func_name):
    """The C type of ``as_type<T>(...)``, or None for another call."""
    if not is_as_type_call(func_name) or generator.is_user_defined_function(func_name):
        return None
    return generator.map_type(_target_type(func_name))


def lower_as_type(generator, func_name, raw_args, args, refusal, vector_helper):
    """C++ for ``as_type<T>(arg)``; ``vector_helper(source_info, result_info)`` names a vector bit-cast helper."""
    target_text = _target_type(func_name)
    if len(args) != 1:
        raise refusal(
            [f"bit reinterpretation: {func_name} takes one operand, not {len(args)}"]
        )
    source_type = generator.expression_result_type(raw_args[0])
    source = (
        generator.map_type(generator.type_name_string(source_type))
        if source_type is not None
        else None
    )
    target = generator.map_type(target_text)
    if source is not None and source == target:
        return args[0]
    source_scalar, target_scalar = _SCALARS.get(source), _SCALARS.get(target)
    if (
        source_scalar is not None
        and target_scalar is not None
        and source_scalar[0] == target_scalar[0]
    ):
        size = source_scalar[0]
        return _INTRINSICS[size][(source_scalar[1], target_scalar[1])].format(args[0])
    source_info = generator.vector_type_info(source) if source is not None else None
    target_info = generator.vector_type_info(target)
    if (
        source_info is not None
        and target_info is not None
        and len(source_info["components"]) == len(target_info["components"])
        and {source_info["component_type"], target_info["component_type"]}
        <= {"float", "int", "uint"}
    ):
        return f"{vector_helper(source_info, target_info)}({args[0]})"
    described = source if source is not None else "an operand of unknown type"
    raise refusal(
        [
            f"bit reinterpretation: as_type<{target_text}> of {described}: only a scalar of the same size or a "
            "vector of 32-bit components with as many components can be reinterpreted"
        ]
    )
