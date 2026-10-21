"""HLSL lowering of the CrossGL wave operations Shader Model 6 has no intrinsic for.

``crosstl.translator.wave_operations`` defines the operations. Each lowers to Shader Model 6.0 wave intrinsics:

* Ballot arithmetic works on the ``uint4`` mask ``WaveActiveBallot`` returns, keeping the bits below
  ``WaveGetLaneCount()``, and the GLSL lane masks (``gl_SubgroupEqMask``, ...) are built from ``WaveGetLaneIndex()``.
* Scans with min, max and bitwise operators and clustered reductions read every lane with ``WaveReadLaneAt``
  in a loop over ``WaveGetLaneCount()`` lanes and combine the values of the active lanes in scope. The loop bound
  and lane index are wave-uniform, so every active lane executes each read, and the ``WaveActiveBallot(true)``
  mask taken at the call keeps out the values of inactive lanes. An exclusive scan starts from the operator's
  identity, as ``OpGroupNonUniform*`` defines it.
* ``WaveActiveBitAnd``, ``WaveActiveBitOr`` and ``WaveActiveBitXor`` of ``bool`` values are the logical reductions.
"""

from __future__ import annotations

import re

from ..wave_operations import (
    SUBGROUP_LANE_MASK_BUILTINS,
    WAVE_BALLOT_BIT_OPERATIONS,
    WAVE_BITWISE_OPERATORS,
    WAVE_EXTENDED_ARITIES,
    cluster_size_value,
    wave_operation_operator,
)
from .array_utils import evaluate_literal_int_expression

__all__ = [
    "emit_hlsl_wave_extension_helpers",
    "hlsl_subgroup_lane_mask_expression",
    "hlsl_wave_extension_call",
]

_FLOAT_COMPONENTS = {"float", "half", "float16_t", "min16float"}
_LANE_LOOP_COMPONENTS = _FLOAT_COMPONENTS | {"double", "int", "uint"}

_IDENTITIES = {
    ("min", "float"): "asfloat(0x7f800000u)",
    ("min", "double"): "asdouble(0u, 0x7ff00000u)",
    ("min", "int"): "2147483647",
    ("min", "uint"): "0xffffffffu",
    ("max", "float"): "asfloat(0xff800000u)",
    ("max", "double"): "asdouble(0u, 0xfff00000u)",
    ("max", "int"): "(-2147483647 - 1)",
    ("max", "uint"): "0u",
    ("and", "int"): "-1",
    ("and", "uint"): "0xffffffffu",
    ("or", "int"): "0",
    ("or", "uint"): "0u",
    ("xor", "int"): "0",
    ("xor", "uint"): "0u",
}

_COMBINE = {
    "sum": "result + other",
    "product": "result * other",
    "min": "min(result, other)",
    "max": "max(result, other)",
    "and": "result & other",
    "or": "result | other",
    "xor": "result ^ other",
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
    helpers = getattr(codegen, "hlsl_wave_extension_helpers", None)
    if helpers is None:
        helpers = {}
        codegen.hlsl_wave_extension_helpers = helpers
    return helpers


def _require(codegen, name, code):
    _helpers(codegen).setdefault(name, code)
    return name


def _error(operation, reason):
    return ValueError(f"DirectX wave intrinsic '{operation}' {reason}")


def _software_subgroup_error(codegen, operation):
    return codegen.hlsl_software_subgroup_error(
        f"DirectX software subgroups have no lowering of '{operation}'",
        operation=operation,
        reason="unsupported-operation",
    )


def _type_suffix(mapped_type):
    return re.sub(r"[^A-Za-z0-9_]", "_", mapped_type)


def _component_kind(component):
    if component in _FLOAT_COMPONENTS:
        return "float"
    return component


def _require_lanes_below(codegen):
    _require(
        codegen,
        _WORD_BELOW,
        f"uint {_WORD_BELOW}(uint lane, uint base) {{\n"
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
        f"uint4 {_LANES_BELOW}(uint lane) {{\n"
        f"    return uint4({_WORD_BELOW}(lane, 0u), {_WORD_BELOW}(lane, 32u),\n"
        f"        {_WORD_BELOW}(lane, 64u), {_WORD_BELOW}(lane, 96u));\n"
        "}\n\n",
    )


def _require_popcount(codegen):
    return _require(
        codegen,
        _POPCOUNT,
        f"uint {_POPCOUNT}(uint4 mask) {{\n"
        "    uint4 counts = countbits(mask);\n"
        "    return counts.x + counts.y + counts.z + counts.w;\n"
        "}\n\n",
    )


def _require_bit_extract(codegen):
    return _require(
        codegen,
        _BIT_EXTRACT,
        f"bool {_BIT_EXTRACT}(uint4 mask, uint index) {{\n"
        "    return index < 128u && ((mask[index >> 5u] >> (index & 31u)) & 1u) != 0u;\n"
        "}\n\n",
    )


def _require_find(codegen, lowest):
    lanes_below = _require_lanes_below(codegen)
    name = "__crossgl_ballot_find_lsb" if lowest else "__crossgl_ballot_find_msb"
    words = (("x", 0), ("y", 32), ("z", 64), ("w", 96))
    if not lowest:
        words = tuple(reversed(words))
    intrinsic = "firstbitlow" if lowest else "firstbithigh"
    body = "".join(
        f"    if (mask.{component} != 0u) {{\n"
        f"        return {base}u + {intrinsic}(mask.{component});\n"
        "    }\n"
        for component, base in words
    )
    return _require(
        codegen,
        name,
        f"uint {name}(uint4 mask) {{\n"
        f"    mask &= {lanes_below}(WaveGetLaneCount());\n"
        f"{body}"
        "    return 0xffffffffu;\n"
        "}\n\n",
    )


def _ballot_mask_argument(codegen, operation, argument):
    mapped = codegen.map_type(codegen.expression_result_type(argument))
    if mapped not in {None, "uint4"}:
        raise _error(operation, f"requires a uint4 ballot mask, got {mapped}")
    return codegen.generate_expression_with_expected(argument, "uint4")


def _ballot_call(codegen, operation, args):
    mask = _ballot_mask_argument(codegen, operation, args[0])
    if operation in {
        "WaveBallotBitCount",
        "WaveBallotInclusiveBitCount",
        "WaveBallotExclusiveBitCount",
    }:
        lanes_below = _require_lanes_below(codegen)
        popcount = _require_popcount(codegen)
        bound = {
            "WaveBallotBitCount": "WaveGetLaneCount()",
            "WaveBallotInclusiveBitCount": "WaveGetLaneIndex() + 1u",
            "WaveBallotExclusiveBitCount": "WaveGetLaneIndex()",
        }[operation]
        return f"{popcount}(({mask}) & {lanes_below}({bound}))"
    if operation in {"WaveBallotFindLSB", "WaveBallotFindMSB"}:
        helper = _require_find(codegen, operation == "WaveBallotFindLSB")
        return f"{helper}({mask})"
    extract = _require_bit_extract(codegen)
    if operation == "WaveInverseBallot":
        return f"{extract}({mask}, WaveGetLaneIndex())"
    index = codegen.generate_expression_with_expected(args[1], "uint")
    return f"{extract}({mask}, uint({index}))"


def _value_type(codegen, operation, argument):
    component, mapped = codegen.hlsl_wave_value_component_type(argument)
    if mapped is None:
        raise _error(operation, "requires a statically known value type")
    if component is None:
        raise _error(operation, f"requires a scalar or vector value, got {mapped}")
    return component, mapped


def _bool_as_uint_type(mapped):
    return "uint" + mapped[len("bool") :]


def _lane_loop_helper(codegen, operation, operator, scope, mapped, component):
    kind = _component_kind(component)
    name = f"__crossgl_{operation}_{_type_suffix(mapped)}"
    if scope == "exclusive":
        identity = _IDENTITIES.get((operator, kind))
        if identity is None:
            raise _error(operation, f"has no {operator} identity for {mapped}")
        initial = f"({mapped})({identity})"
        condition = "source < lane"
    elif scope == "inclusive":
        initial = "value"
        condition = "source < lane"
    else:
        initial = "value"
        condition = "source != lane && source / clusterSize == lane / clusterSize"
    parameters = f"{mapped} value"
    if scope == "cluster":
        parameters += ", uint clusterSize"
    return _require(
        codegen,
        name,
        f"{mapped} {name}({parameters}) {{\n"
        "    uint lane = WaveGetLaneIndex();\n"
        "    uint4 active = WaveActiveBallot(true);\n"
        f"    {mapped} result = {initial};\n"
        "    for (uint source = 0u; source < WaveGetLaneCount(); ++source) {\n"
        f"        {mapped} other = WaveReadLaneAt(value, source);\n"
        "        bool member = ((active[source >> 5u] >> (source & 31u)) & 1u) != 0u;\n"
        f"        if (member && {condition}) {{\n"
        f"            result = {_COMBINE[operator]};\n"
        "        }\n"
        "    }\n"
        "    return result;\n"
        "}\n\n",
    )


def _cluster_size_argument(codegen, operation, argument):
    literal = evaluate_literal_int_expression(argument)
    if literal is not None and cluster_size_value(literal) is None:
        raise _error(operation, f"cluster size must be a power of two, got {literal}")
    return codegen.generate_expression_with_expected(argument, "uint")


def _scan_or_cluster_call(codegen, operation, args):
    operator, scope = wave_operation_operator(operation)
    component, mapped = _value_type(codegen, operation, args[0])
    boolean = component == "bool"
    if boolean and operator not in WAVE_BITWISE_OPERATORS:
        raise _error(operation, f"requires a numeric value, got {mapped}")
    if not boolean and component not in _LANE_LOOP_COMPONENTS:
        raise _error(operation, f"does not support {mapped} values")
    if (
        not boolean
        and operator in WAVE_BITWISE_OPERATORS
        and _component_kind(component) == "float"
    ):
        raise _error(operation, f"requires an integer or bool value, got {mapped}")
    value = codegen.generate_expression_with_expected(args[0], mapped)
    extra = ""
    if scope == "cluster":
        extra = f", {_cluster_size_argument(codegen, operation, args[1])}"
    if boolean:
        carrier = _bool_as_uint_type(mapped)
        helper = _lane_loop_helper(codegen, operation, operator, scope, carrier, "uint")
        return f"({helper}(({carrier})({value}){extra}) != 0u)"
    helper = _lane_loop_helper(codegen, operation, operator, scope, mapped, component)
    return f"{helper}({value}{extra})"


def _boolean_reduction_call(codegen, operation, args, operator, mapped):
    value = codegen.generate_expression_with_expected(args[0], mapped)
    if mapped == "bool":
        if operator == "and":
            return f"WaveActiveAllTrue({value})"
        if operator == "or":
            return f"WaveActiveAnyTrue({value})"
        return f"((WaveActiveCountBits({value}) & 1u) != 0u)"
    count = codegen.value_component_count(mapped)
    if not count:
        raise _error(operation, f"does not support {mapped} values")
    intrinsic = {
        "and": "WaveActiveAllTrue(value.{c})",
        "or": "WaveActiveAnyTrue(value.{c})",
        "xor": "((WaveActiveCountBits(value.{c}) & 1u) != 0u)",
    }[operator]
    lanes = ", ".join(intrinsic.format(c=c) for c in "xyzw"[:count])
    name = _require(
        codegen,
        f"__crossgl_{operation}_{mapped}",
        f"{mapped} __crossgl_{operation}_{mapped}({mapped} value) {{\n"
        f"    return {mapped}({lanes});\n"
        "}\n\n",
    )
    return f"{name}({value})"


def hlsl_wave_extension_call(codegen, operation, args):
    """HLSL for ``operation``, or ``None`` when the generator's Shader Model 6 path owns it."""
    expected = WAVE_EXTENDED_ARITIES.get(operation)
    if expected is not None:
        if len(args) != expected:
            raise _error(operation, f"requires {expected} argument(s), got {len(args)}")
        if codegen.software_subgroup_width is not None:
            raise _software_subgroup_error(codegen, operation)
        codegen.validate_directx_wave_intrinsic_profile(operation)
        if operation in WAVE_BALLOT_BIT_OPERATIONS:
            return _ballot_call(codegen, operation, args)
        return _scan_or_cluster_call(codegen, operation, args)
    operator = _BOOLEAN_REDUCTIONS.get(operation)
    if operator is None or len(args) != 1:
        return None
    component, mapped = codegen.hlsl_wave_value_component_type(args[0])
    if component != "bool" or codegen.software_subgroup_width is not None:
        return None
    codegen.validate_directx_wave_intrinsic_profile(operation)
    return _boolean_reduction_call(codegen, operation, args, operator, mapped)


def hlsl_subgroup_lane_mask_expression(codegen, name):
    """HLSL for a GLSL lane-mask built-in such as ``gl_SubgroupEqMask``, or ``None`` for other names."""
    relation = SUBGROUP_LANE_MASK_BUILTINS.get(name)
    if relation is None:
        return None
    if codegen.software_subgroup_width is not None:
        raise _software_subgroup_error(codegen, name)
    codegen.validate_directx_wave_intrinsic_profile(name)
    below = _require_lanes_below(codegen)
    lane = f"{below}(WaveGetLaneIndex())"
    through_lane = f"{below}(WaveGetLaneIndex() + 1u)"
    in_subgroup = f"{below}(WaveGetLaneCount())"
    return {
        "==": f"({through_lane} & ~{lane})",
        ">=": f"(~{lane} & {in_subgroup})",
        ">": f"(~{through_lane} & {in_subgroup})",
        "<=": through_lane,
        "<": lane,
    }[relation]


def emit_hlsl_wave_extension_helpers(codegen):
    """The helper functions the lowered operations call, in the order they were first needed."""
    return "".join(_helpers(codegen).values())
