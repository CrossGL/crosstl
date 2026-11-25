"""CrossGL wave operations beyond Shader Model 6.0 written with the wave functions of Slang's core module.

Slang v2026.20 (``hlsl.meta.slang``) declares, besides HLSL's Shader Model 6.0 wave set:

- ``QuadAll`` and ``QuadAny`` (SPIR-V ``OpGroupNonUniformQuadAllKHR``/``QuadAnyKHR``), ``WaveActiveAllEqual``,
  ``WaveActiveCountBits`` and ``WavePrefixCountBits``, with CrossGL's operands and results;
- ``WavePrefix{Min,Max,BitAnd,BitOr,BitXor}``, exclusive scans like its ``WavePrefixSum``: they lower to
  ``OpGroupNonUniform*`` ``ExclusiveScan``, so they are CrossGL's ``WavePrefixExclusive*``;
- ``WaveClustered{Sum,Product,Min,Max,BitAnd,BitOr,BitXor}(value, constexpr uint clusterSize)``, the
  ``ClusteredReduce`` group operations.

Slang declares ``WaveMultiPrefixCountBits`` for HLSL and CUDA only. Over the lanes of its partition mask it counts
the lower lanes whose predicate is true, which is the ballot of the predicate masked by the partition and by
``WaveGetLaneLtMask()`` (SPIR-V ``SubgroupLtMask``), counted; a helper function writes that.

Slang has no inclusive scan. CrossGL's ``WavePrefixInclusive*`` is the exclusive scan combined with the lane's own
value, as the HLSL generator writes it, in a helper function so the operand is evaluated once.

The bitwise operations take integer operands in Slang. A ``bool`` operand of CrossGL's, where they are the logical
operators, is scanned or reduced as a ``uint`` 0 or 1 and compared with zero: the identities of AND (all bits set),
OR and XOR (zero) are then ``true``, ``false`` and ``false``, as the logical operators' are.
"""

from __future__ import annotations

import re

from ..wave_operations import cluster_size_value
from .array_utils import evaluate_literal_int_expression

__all__ = [
    "SLANG_EXTENDED_WAVE_OPERATIONS",
    "generate_slang_extended_wave_operation",
]

# CrossGL operation -> (Slang function, operand count, result): "bool", "uint" or "value" (the operand's type).
_DIRECT = {
    "QuadAll": ("QuadAll", 1, "bool"),
    "QuadAny": ("QuadAny", 1, "bool"),
    "WaveActiveAllEqual": ("WaveActiveAllEqual", 1, "bool"),
    "WaveActiveCountBits": ("WaveActiveCountBits", 1, "uint"),
    "WavePrefixCountBits": ("WavePrefixCountBits", 1, "uint"),
    "WavePrefixExclusiveMin": ("WavePrefixMin", 1, "value"),
    "WavePrefixExclusiveMax": ("WavePrefixMax", 1, "value"),
    "WavePrefixExclusiveBitAnd": ("WavePrefixBitAnd", 1, "value"),
    "WavePrefixExclusiveBitOr": ("WavePrefixBitOr", 1, "value"),
    "WavePrefixExclusiveBitXor": ("WavePrefixBitXor", 1, "value"),
    "WaveClusteredSum": ("WaveClusteredSum", 2, "value"),
    "WaveClusteredProduct": ("WaveClusteredProduct", 2, "value"),
    "WaveClusteredMin": ("WaveClusteredMin", 2, "value"),
    "WaveClusteredMax": ("WaveClusteredMax", 2, "value"),
    "WaveClusteredBitAnd": ("WaveClusteredBitAnd", 2, "value"),
    "WaveClusteredBitOr": ("WaveClusteredBitOr", 2, "value"),
    "WaveClusteredBitXor": ("WaveClusteredBitXor", 2, "value"),
}
# Inclusive scan -> (Slang exclusive scan, how the lane's own value joins it).
_INCLUSIVE = {
    "WavePrefixInclusiveSum": ("WavePrefixSum", "{scan} + {value}"),
    "WavePrefixInclusiveProduct": ("WavePrefixProduct", "{scan} * {value}"),
    "WavePrefixInclusiveMin": ("WavePrefixMin", "min({scan}, {value})"),
    "WavePrefixInclusiveMax": ("WavePrefixMax", "max({scan}, {value})"),
    "WavePrefixInclusiveBitAnd": ("WavePrefixBitAnd", "{scan} & {value}"),
    "WavePrefixInclusiveBitOr": ("WavePrefixBitOr", "{scan} | {value}"),
    "WavePrefixInclusiveBitXor": ("WavePrefixBitXor", "{scan} ^ {value}"),
}
_MULTI_PREFIX_COUNT_BITS = "WaveMultiPrefixCountBits"
SLANG_EXTENDED_WAVE_OPERATIONS = (
    frozenset(_DIRECT) | frozenset(_INCLUSIVE) | {_MULTI_PREFIX_COUNT_BITS}
)

_BITWISE = frozenset(
    name
    for name, (function, _arity, _result) in {
        **_DIRECT,
        **{name: (scan, 1, "value") for name, (scan, _) in _INCLUSIVE.items()},
    }.items()
    if re.search(r"Bit(?:And|Or|Xor)$", function)
)
_PREDICATES = frozenset(
    {"QuadAll", "QuadAny", "WaveActiveCountBits", "WavePrefixCountBits"}
)
_NUMERIC = re.compile(
    r"(?:bool|int|uint|float|double|half|int8_t|uint8_t|int16_t|uint16_t|int64_t|uint64_t)[234]?"
    r"|vector<\s*\w+\s*,\s*[234]\s*>"
)
_INTEGER = re.compile(
    r"(?:int|uint|int8_t|uint8_t|int16_t|uint16_t|int64_t|uint64_t)[234]?"
    r"|vector<\s*u?int(?:8|16|64)_t\s*,\s*[234]\s*>"
)
_BOOL = re.compile(r"bool([234]?)")


def _value_type(codegen, argument):
    mapped = codegen.slang_wave_argument_mapped_type(argument)
    return mapped.strip() if isinstance(mapped, str) else None


def _refuse(codegen, operation, reason):
    return codegen.unsupported_slang_wave_op_expression(operation, reason)


def _bitwise_operand(codegen, operation, value_type, rendered):
    """The operand of a bitwise operation and the conversion of its result, or a refusal reason."""
    boolean = _BOOL.fullmatch(value_type)
    if boolean:
        width = boolean.group(1)
        return f"uint{width}({rendered})", f"({{call}} != uint{width}(0))", None
    if not _INTEGER.fullmatch(value_type):
        return (
            None,
            None,
            f"value must be an integer or bool scalar or vector, got {value_type}",
        )
    return rendered, "{call}", None


def _inclusive_helper(codegen, operation, value_type, operand_type, result_form):
    scan, combine = _INCLUSIVE[operation]
    suffix = re.sub(r"\W+", "_", value_type).strip("_")
    name = codegen.helper_function_name(f"cgl_{operation}_{suffix}")
    if operand_type == value_type:
        body = combine.format(scan=f"{scan}(value)", value="value")
        source = f"{value_type} {name}({value_type} value)\n{{\n    return {body};\n}}"
    else:
        # A bool scan runs on its 0/1 uint operand and converts back.
        body = combine.format(scan=f"{scan}(bits)", value="bits")
        result = result_form.format(call=f"({body})")
        source = (
            f"{value_type} {name}({value_type} value)\n{{\n"
            f"    {operand_type} bits = {operand_type}(value);\n"
            f"    return {result};\n}}"
        )
    codegen.register_helper_function(name, source)
    return name


def _multi_prefix_count_bits(codegen, arguments):
    operation = _MULTI_PREFIX_COUNT_BITS
    if len(arguments) != 2:
        return _refuse(codegen, operation, f"expects 2 arguments, got {len(arguments)}")
    predicate_type = _value_type(codegen, arguments[0])
    if predicate_type != "bool":
        return _refuse(
            codegen,
            operation,
            f"predicate must be scalar bool, got {predicate_type or 'an unknown type'}",
        )
    mask_type = _value_type(codegen, arguments[1])
    if mask_type != "uint4":
        return _refuse(
            codegen,
            operation,
            f"partition mask must be uint4, got {mask_type or 'an unknown type'}",
        )
    name = codegen.helper_function_name("cgl_WaveMultiPrefixCountBits")
    codegen.register_helper_function(
        name,
        f"uint {name}(bool value, uint4 mask)\n"
        "{\n"
        "    uint4 bits = WaveActiveBallot(value) & mask & WaveGetLaneLtMask();\n"
        "    return countbits(bits.x) + countbits(bits.y) + countbits(bits.z) + countbits(bits.w);\n"
        "}",
    )
    rendered = ", ".join(
        codegen.generate_expression_without_expected(arg) for arg in arguments
    )
    call = f"{name}({rendered})"
    conversion = codegen.slang_wave_result_conversion("uint")
    if conversion is None:
        reason = codegen.slang_wave_target_type_rejection_reason("uint")
        if reason is not None:
            return _refuse(codegen, operation, reason)
        return call
    return f"{conversion}({call})"


def generate_slang_extended_wave_operation(codegen, node):
    """Slang source of a CrossGL wave operation of :data:`SLANG_EXTENDED_WAVE_OPERATIONS`, or ``None``."""
    operation = node.operation
    if operation not in SLANG_EXTENDED_WAVE_OPERATIONS:
        return None
    arguments = list(node.arguments)
    if operation == _MULTI_PREFIX_COUNT_BITS:
        return _multi_prefix_count_bits(codegen, arguments)
    arity = _DIRECT[operation][1] if operation in _DIRECT else 1
    if len(arguments) != arity:
        return _refuse(
            codegen, operation, f"expects {arity} arguments, got {len(arguments)}"
        )
    value_type = _value_type(codegen, arguments[0])
    if value_type is None:
        return _refuse(codegen, operation, "value has an unknown type")
    if operation in _PREDICATES:
        if value_type != "bool":
            return _refuse(
                codegen, operation, f"predicate must be scalar bool, got {value_type}"
            )
    elif not _NUMERIC.fullmatch(value_type):
        return _refuse(
            codegen, operation, f"value must be a scalar or vector, got {value_type}"
        )
    rendered = codegen.generate_expression_without_expected(arguments[0])
    operand, result_form = rendered, "{call}"
    if operation in _BITWISE:
        operand, result_form, reason = _bitwise_operand(
            codegen, operation, value_type, rendered
        )
        if reason is not None:
            return _refuse(codegen, operation, reason)
    elif (
        operation not in _PREDICATES
        and operation != "WaveActiveAllEqual"
        and _BOOL.fullmatch(value_type)
    ):
        return _refuse(codegen, operation, f"value must be numeric, got {value_type}")
    if operation in _INCLUSIVE:
        operand_type = value_type
        boolean = _BOOL.fullmatch(value_type)
        if boolean and operation in _BITWISE:
            operand_type = f"uint{boolean.group(1)}"
        name = _inclusive_helper(
            codegen, operation, value_type, operand_type, result_form
        )
        call, result_type = f"{name}({rendered})", value_type
    else:
        function, _arity, result = _DIRECT[operation]
        operands = [operand]
        if arity == 2:
            size = cluster_size_value(evaluate_literal_int_expression(arguments[1]))
            if size is None:
                return _refuse(
                    codegen, operation, "requires a constant power-of-two cluster size"
                )
            operands.append(f"{size}u")
        call = result_form.format(call=f"{function}({', '.join(operands)})")
        result_type = value_type if result == "value" else result
    conversion = codegen.slang_wave_result_conversion(result_type)
    if conversion is None:
        reason = codegen.slang_wave_target_type_rejection_reason(result_type)
        if reason is not None:
            return _refuse(codegen, operation, reason)
        return call
    return f"{conversion}({call})"
