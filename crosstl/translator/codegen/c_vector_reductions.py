"""CUDA and HIP spelling of the ``all`` and ``any`` reductions.

CrossGL, HLSL, GLSL and Metal reduce a vector with ``all`` (every component is non-zero) and ``any`` (some
component is non-zero); HLSL also accepts a scalar. Neither CUDA nor HIP has a device function of that meaning: a
boolean vector is written as ``ucharN``, and the only ``all``/``any`` nvcc finds are its legacy warp votes, which
take a ``bool`` and vote across the warp. So a vector operand becomes a call to a generated helper,
``cgl_uchar2_all(value)``, that tests each component once, and a scalar operand becomes ``((x) != 0)``.

An operand whose type the generator cannot name, or a vector type without a helper, is refused rather than left as a
call that means a warp vote or does not compile.
"""

from __future__ import annotations

__all__ = ["REDUCTION_FUNCTIONS", "generate_vector_reduction_call"]

REDUCTION_FUNCTIONS = frozenset({"all", "any"})

_SCALAR_TYPES = frozenset(
    {
        "bool",
        "char",
        "signed char",
        "unsigned char",
        "uchar",
        "short",
        "unsigned short",
        "ushort",
        "int",
        "unsigned int",
        "uint",
        "long",
        "unsigned long",
        "long long",
        "unsigned long long",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
        "float",
        "double",
    }
)
_HALF_TYPES = frozenset({"half", "__half", "float16_t", "f16"})
_HALF2_TYPES = frozenset({"half2", "__half2"})


def _type_name(codegen, vtype):
    if vtype is None:
        return None
    if isinstance(vtype, str):
        return vtype.strip()
    return codegen.convert_type_node_to_string(vtype)


def _non_zero(component_type, value):
    # __half compares with __half only; converting it to float is exact.
    if component_type in _HALF_TYPES:
        return f"(static_cast<float>({value}) != 0.0f)"
    return f"({value} != 0)"


def _require_helper(codegen, func_name, vector_info):
    vector_type = vector_info["type"]
    helper_name = f"cgl_{vector_type.removeprefix('cgl_')}_{func_name}"
    if helper_name in codegen.helper_functions:
        return helper_name
    joiner = " && " if func_name == "all" else " || "
    tests = joiner.join(
        _non_zero(vector_info["component_type"], f"value.{component}")
        for component in vector_info["components"]
    )
    codegen.helper_functions[helper_name] = (
        f"__device__ inline bool {helper_name}({vector_type} value)\n"
        "{\n"
        f"    return {tests};\n"
        "}"
    )
    return helper_name


def generate_vector_reduction_call(codegen, target, func_name, raw_args, args):
    """The ``target`` spelling of ``all(x)`` or ``any(x)``; ``None`` for any other call."""
    if func_name not in REDUCTION_FUNCTIONS:
        return None
    if len(raw_args) != 1 or len(args) != 1:
        raise ValueError(
            f"{target} target: {func_name} takes one argument; got {len(raw_args)}"
        )
    operand_type = codegen.expression_result_type(raw_args[0])
    vector_info = codegen.vector_type_info(operand_type)
    type_name = _type_name(codegen, operand_type)
    mapped = codegen.map_vector_arithmetic_type(type_name) if type_name else None
    if vector_info is None and mapped in _HALF2_TYPES:
        # Both runtimes write a two-lane half vector as their native __half2, whose lanes are x and y.
        vector_info = {
            "type": mapped,
            "component_type": "half",
            "components": ("x", "y"),
        }
    if vector_info is not None:
        return f"{_require_helper(codegen, func_name, vector_info)}({args[0]})"
    for scalar in (type_name, mapped):
        if scalar in _SCALAR_TYPES or scalar in _HALF_TYPES:
            return _non_zero(scalar, f"({args[0]})")
    raise ValueError(
        f"{target} target cannot lower {func_name}() on an operand of type "
        f"{type_name or 'unknown'}: it has no device {func_name} reduction, and "
        f"the {func_name} it does have is a warp vote"
    )
