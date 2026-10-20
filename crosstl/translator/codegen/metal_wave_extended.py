"""Metal lowering of the CrossGL wave operations MSL has no SIMD-group function for.

``crosstl.translator.wave_operations`` defines the operations. They lower to SIMD-group functions the way the
HLSL lowering (``hlsl_wave_extended``) uses Shader Model 6 wave intrinsics:

* Ballot arithmetic and the GLSL lane masks work on the ``uint4`` ballot the generator builds from
  ``simd_ballot``, keeping the bits below ``threads_per_simdgroup``.
* Scans with min, max and bitwise operators and clustered reductions read every lane with ``simd_shuffle`` in a
  loop over the SIMD-group's lanes and combine the values of the active lanes in scope. The loop bound and lane
  index are uniform, so every active lane executes each read, and the ``simd_ballot(true)`` mask taken at the call
  keeps out inactive lanes. Clusters of four lanes are quad-groups and use the ``quad_*`` reductions.
* ``WaveActiveBitAnd``, ``WaveActiveBitOr`` and ``WaveActiveBitXor`` of ``bool`` values are the logical reductions.

The lane index and count come from the ``thread_index_in_simdgroup`` and ``threads_per_simdgroup`` values the
generator threads through every function that needs them.
"""

from __future__ import annotations

import re

from ..ast import FunctionCallNode, WaveOpNode
from ..wave_operations import (
    SUBGROUP_LANE_MASK_BUILTINS,
    WAVE_BALLOT_BIT_OPERATIONS,
    WAVE_BITWISE_OPERATORS,
    WAVE_EXTENDED_ARITIES,
    WAVE_EXTENDED_OPERATIONS,
    cluster_size_value,
    wave_operation_operator,
)
from .array_utils import evaluate_literal_int_expression

__all__ = [
    "emit_metal_wave_extension_helpers",
    "metal_subgroup_lane_mask_expression",
    "metal_wave_extension_call",
    "metal_wave_extension_lane_requirements",
]

_FLOAT_COMPONENTS = {"float", "half"}
_LANE_LOOP_COMPONENTS = _FLOAT_COMPONENTS | {"int", "uint", "short", "ushort"}

_COMBINE = {
    "sum": "result + other",
    "product": "result * other",
    "min": "min(result, other)",
    "max": "max(result, other)",
    "and": "result & other",
    "or": "result | other",
    "xor": "result ^ other",
}

_QUAD_REDUCTIONS = {
    "sum": "quad_sum",
    "product": "quad_product",
    "min": "quad_min",
    "max": "quad_max",
    "and": "quad_and",
    "or": "quad_or",
    "xor": "quad_xor",
}

_BOOLEAN_REDUCTIONS = {
    "WaveActiveBitAnd": "and",
    "WaveActiveBitOr": "or",
    "WaveActiveBitXor": "xor",
}

_WORD_BELOW = "__crossgl_ballot_word_below"
_LANES_BELOW = "__crossgl_ballot_lanes_below"
_POPCOUNT = "__crossgl_ballot_popcount"
_BIT_EXTRACT = "__crossgl_ballot_bit_extract"


def _helpers(codegen):
    helpers = getattr(codegen, "metal_wave_extension_helpers", None)
    if helpers is None:
        helpers = {}
        codegen.metal_wave_extension_helpers = helpers
    return helpers


def _require(codegen, name, code):
    _helpers(codegen).setdefault(name, code)
    return name


def _type_suffix(mapped_type):
    return re.sub(r"[^A-Za-z0-9_]", "_", mapped_type)


def _refuse(codegen, operation, arguments, reason):
    return codegen.metal_wave_diagnostic_expression(operation, arguments, reason)


def metal_wave_extension_lane_requirements(node):
    """The lane values (``WaveGetLaneIndex``/``WaveGetLaneCount``) the lowering of ``node`` reads."""
    if isinstance(node, WaveOpNode):
        operation = getattr(node, "operation", None)
    elif isinstance(node, FunctionCallNode):
        name = getattr(node, "name", None)
        operation = getattr(name, "name", name)
    elif type(node).__name__ in {"IdentifierNode", "VariableNode"}:
        operation = None
        if getattr(node, "name", None) in SUBGROUP_LANE_MASK_BUILTINS:
            return {"WaveGetLaneIndex", "WaveGetLaneCount"}
    else:
        operation = None
    if operation in WAVE_EXTENDED_OPERATIONS:
        return {"WaveGetLaneIndex", "WaveGetLaneCount"}
    return set()


def _lane_values(codegen):
    return (
        codegen.current_metal_wave_lane_index_parameter,
        codegen.current_metal_wave_lane_count_parameter,
    )


def _require_lanes_below(codegen):
    _require(
        codegen,
        _WORD_BELOW,
        f"static uint {_WORD_BELOW}(uint lane, uint base) {{\n"
        "    if (lane <= base) {\n"
        "        return 0u;\n"
        "    }\n"
        "    uint count = lane - base;\n"
        "    return count >= 32u ? 0xffffffffu : ((1u << count) - 1u);\n"
        "}\n\n",
    )
    return _require(
        codegen,
        _LANES_BELOW,
        f"static uint4 {_LANES_BELOW}(uint lane) {{\n"
        f"    return uint4({_WORD_BELOW}(lane, 0u), {_WORD_BELOW}(lane, 32u),\n"
        f"        {_WORD_BELOW}(lane, 64u), {_WORD_BELOW}(lane, 96u));\n"
        "}\n\n",
    )


def _require_popcount(codegen):
    return _require(
        codegen,
        _POPCOUNT,
        f"static uint {_POPCOUNT}(uint4 mask) {{\n"
        "    uint4 counts = popcount(mask);\n"
        "    return counts.x + counts.y + counts.z + counts.w;\n"
        "}\n\n",
    )


def _require_bit_extract(codegen):
    return _require(
        codegen,
        _BIT_EXTRACT,
        f"static bool {_BIT_EXTRACT}(uint4 mask, uint index) {{\n"
        "    return index < 128u && ((mask[index >> 5u] >> (index & 31u)) & 1u) != 0u;\n"
        "}\n\n",
    )


def _require_find(codegen, lowest):
    lanes_below = _require_lanes_below(codegen)
    name = "__crossgl_ballot_find_lsb" if lowest else "__crossgl_ballot_find_msb"
    words = (("x", 0), ("y", 32), ("z", 64), ("w", 96))
    if not lowest:
        words = tuple(reversed(words))
    position = "ctz(mask.{c})" if lowest else "(31u - clz(mask.{c}))"
    body = "".join(
        f"    if (mask.{component} != 0u) {{\n"
        f"        return {base}u + {position.format(c=component)};\n"
        "    }\n"
        for component, base in words
    )
    return _require(
        codegen,
        name,
        f"static uint {name}(uint4 mask, uint laneCount) {{\n"
        f"    mask &= {lanes_below}(laneCount);\n"
        f"{body}"
        "    return 0xffffffffu;\n"
        "}\n\n",
    )


def _ballot_call(codegen, operation, arguments, lane, count):
    mapped = codegen.map_type(codegen.expression_result_type(arguments[0]))
    if mapped not in {None, "uint4"}:
        return _refuse(
            codegen, operation, arguments, f"requires a uint4 ballot mask, got {mapped}"
        )
    mask = codegen.generate_expression(arguments[0])
    if operation in {
        "WaveBallotBitCount",
        "WaveBallotInclusiveBitCount",
        "WaveBallotExclusiveBitCount",
    }:
        lanes_below = _require_lanes_below(codegen)
        popcount = _require_popcount(codegen)
        bound = {
            "WaveBallotBitCount": count,
            "WaveBallotInclusiveBitCount": f"{lane} + 1u",
            "WaveBallotExclusiveBitCount": lane,
        }[operation]
        return f"{popcount}(({mask}) & {lanes_below}({bound}))"
    if operation in {"WaveBallotFindLSB", "WaveBallotFindMSB"}:
        helper = _require_find(codegen, operation == "WaveBallotFindLSB")
        return f"{helper}({mask}, {count})"
    extract = _require_bit_extract(codegen)
    if operation == "WaveInverseBallot":
        return f"{extract}({mask}, {lane})"
    index = codegen.generate_expression(arguments[1])
    return f"{extract}({mask}, uint({index}))"


def _identity(operator, component, mapped):
    if operator == "min":
        value = (
            f"numeric_limits<{component}>::infinity()"
            if component in _FLOAT_COMPONENTS
            else f"numeric_limits<{component}>::max()"
        )
    elif operator == "max":
        value = (
            f"-numeric_limits<{component}>::infinity()"
            if component in _FLOAT_COMPONENTS
            else f"numeric_limits<{component}>::min()"
        )
    elif operator == "and":
        value = f"{component}(~0u)"
    elif operator in {"or", "xor"}:
        value = f"{component}(0)"
    else:
        return None
    return f"{mapped}({value})"


def _lane_loop_helper(codegen, operation, operator, scope, mapped, component):
    name = f"__crossgl_{operation}_{_type_suffix(mapped)}"
    if scope == "exclusive":
        initial = _identity(operator, component, mapped)
        condition = "source < lane"
    elif scope == "inclusive":
        initial = "value"
        condition = "source < lane"
    else:
        initial = "value"
        condition = "source != lane && source / clusterSize == lane / clusterSize"
    parameters = f"{mapped} value, uint lane, uint laneCount"
    if scope == "cluster":
        parameters += ", uint clusterSize"
    codegen.required_metal_wave_ballot_helper = True
    return _require(
        codegen,
        name,
        f"static {mapped} {name}({parameters}) {{\n"
        "    uint4 active = __crossgl_metal_wave_ballot(true);\n"
        f"    {mapped} result = {initial};\n"
        "    for (uint source = 0u; source < laneCount; ++source) {\n"
        f"        {mapped} other = simd_shuffle(value, ushort(source));\n"
        "        bool member = ((active[source >> 5u] >> (source & 31u)) & 1u) != 0u;\n"
        f"        if (member && {condition}) {{\n"
        f"            result = {_COMBINE[operator]};\n"
        "        }\n"
        "    }\n"
        "    return result;\n"
        "}\n\n",
    )


def _value_type(codegen, argument):
    mapped, component, array_suffix = codegen.metal_wave_argument_mapped_type(argument)
    if mapped is None or array_suffix:
        return None, mapped
    return component or mapped, mapped


def _scan_or_cluster_call(codegen, operation, arguments, lane, count):
    operator, scope = wave_operation_operator(operation)
    component, mapped = _value_type(codegen, arguments[0])
    if component is None:
        return _refuse(
            codegen,
            operation,
            arguments,
            f"requires a scalar or vector value, got {mapped or 'an unknown type'}",
        )
    boolean = component == "bool"
    if boolean and operator not in WAVE_BITWISE_OPERATORS:
        return _refuse(
            codegen, operation, arguments, f"requires a numeric value, got {mapped}"
        )
    if not boolean and component not in _LANE_LOOP_COMPONENTS:
        return _refuse(
            codegen, operation, arguments, f"does not support {mapped} values"
        )
    if (
        not boolean
        and operator in WAVE_BITWISE_OPERATORS
        and component in _FLOAT_COMPONENTS
    ):
        return _refuse(
            codegen,
            operation,
            arguments,
            f"requires an integer or bool value, got {mapped}",
        )
    value = codegen.generate_expression(arguments[0])
    cluster = ""
    if scope == "cluster":
        literal = evaluate_literal_int_expression(arguments[1])
        if literal is not None and cluster_size_value(literal) is None:
            return _refuse(
                codegen,
                operation,
                arguments,
                f"cluster size must be a power of two, got {literal}",
            )
        if literal == 1:
            return f"({value})"
        if literal == 4:
            if boolean:
                carrier = "uint" + mapped[len("bool") :]
                return f"({_QUAD_REDUCTIONS[operator]}({carrier}({value})) != 0u)"
            return f"{_QUAD_REDUCTIONS[operator]}({value})"
        cluster = f", uint({codegen.generate_expression(arguments[1])})"
    if boolean:
        carrier = "uint" + mapped[len("bool") :]
        helper = _lane_loop_helper(codegen, operation, operator, scope, carrier, "uint")
        return f"({helper}({carrier}({value}), {lane}, {count}{cluster}) != 0u)"
    helper = _lane_loop_helper(codegen, operation, operator, scope, mapped, component)
    return f"{helper}({value}, {lane}, {count}{cluster})"


def _boolean_reduction_call(codegen, operation, arguments, operator, mapped):
    value = codegen.generate_expression(arguments[0])
    if mapped == "bool":
        if operator == "and":
            return f"simd_all({value})"
        if operator == "or":
            return f"simd_any({value})"
        return (
            f"((popcount(simd_vote::vote_t(simd_ballot({value}))) & "
            "simd_vote::vote_t(1u)) != simd_vote::vote_t(0u))"
        )
    count = codegen.value_component_count(mapped)
    if not count or count < 2:
        return _refuse(
            codegen, operation, arguments, f"does not support {mapped} values"
        )
    intrinsic = {
        "and": "simd_all(value.{c})",
        "or": "simd_any(value.{c})",
        "xor": (
            "((popcount(simd_vote::vote_t(simd_ballot(value.{c}))) & "
            "simd_vote::vote_t(1u)) != simd_vote::vote_t(0u))"
        ),
    }[operator]
    lanes = ", ".join(intrinsic.format(c=c) for c in "xyzw"[:count])
    name = _require(
        codegen,
        f"__crossgl_{operation}_{mapped}",
        f"static {mapped} __crossgl_{operation}_{mapped}({mapped} value) {{\n"
        f"    return {mapped}({lanes});\n"
        "}\n\n",
    )
    return f"{name}({value})"


def metal_wave_extension_call(codegen, operation, arguments):
    """MSL for ``operation``, or ``None`` when the generator's own wave lowering owns it."""
    expected = WAVE_EXTENDED_ARITIES.get(operation)
    if expected is not None:
        if len(arguments) != expected:
            return _refuse(
                codegen,
                operation,
                arguments,
                f"expects {expected} arguments, got {len(arguments)}",
            )
        lane, count = _lane_values(codegen)
        if lane is None or count is None:
            return _refuse(
                codegen,
                operation,
                arguments,
                "requires thread_index_in_simdgroup and threads_per_simdgroup values",
            )
        if operation in WAVE_BALLOT_BIT_OPERATIONS:
            return _ballot_call(codegen, operation, arguments, lane, count)
        return _scan_or_cluster_call(codegen, operation, arguments, lane, count)
    operator = _BOOLEAN_REDUCTIONS.get(operation)
    if operator is None or len(arguments) != 1:
        return None
    mapped, component, array_suffix = codegen.metal_wave_argument_mapped_type(
        arguments[0]
    )
    if array_suffix or (component or mapped) != "bool":
        return None
    return _boolean_reduction_call(codegen, operation, arguments, operator, mapped)


def metal_subgroup_lane_mask_expression(codegen, name):
    """MSL for a GLSL lane-mask built-in such as ``gl_SubgroupEqMask``, or ``None`` for other names."""
    relation = SUBGROUP_LANE_MASK_BUILTINS.get(name)
    if relation is None or name in codegen.local_variable_types:
        return None
    lane, count = _lane_values(codegen)
    if lane is None or count is None:
        return codegen.metal_wave_diagnostic_expression(
            name,
            [],
            "requires thread_index_in_simdgroup and threads_per_simdgroup values",
        )
    below = _require_lanes_below(codegen)
    through_lane = f"{below}({lane} + 1u)"
    before_lane = f"{below}({lane})"
    in_subgroup = f"{below}({count})"
    return {
        "==": f"({through_lane} & ~{before_lane})",
        ">=": f"(~{before_lane} & {in_subgroup})",
        ">": f"(~{through_lane} & {in_subgroup})",
        "<=": through_lane,
        "<": before_lane,
    }[relation]


def emit_metal_wave_extension_helpers(codegen):
    """The helper functions the lowered operations call, in the order they were first needed."""
    return "".join(_helpers(codegen).values())
