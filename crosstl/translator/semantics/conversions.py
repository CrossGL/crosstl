"""Conversions of the ``convert_sat`` rule family: resolve the generic forms by operand type.

OpenCL C section 6.4.3.3 defines ``convert_<T>_sat``: the value is clamped to the range of ``T`` and NaN becomes 0.
Sources write the generic hub intrinsics of ``rule_table.CONVERT_SAT_GENERIC`` (``__cgl_convert_int_sat(x)``,
``__cgl_convert_uint_rte(x)``, ...) because the operation depends on the operand type, which only the hub knows.
:func:`resolve` replaces such a call during annotation:

- a binary32 operand is first rounded to an integral value in the named mode (``roundEven``, ``ceil`` or ``floor``;
  ``rtz`` and no mode leave it to the conversion, which truncates), then converted by ``__cgl_sat_ftoi`` or
  ``__cgl_sat_ftou``. A conversion without ``_sat`` saturates too: OpenCL C 6.4.3.3 makes its out-of-range result
  implementation-defined, and saturation is a defined choice where a target's own cast may be undefined;
- a binary16 operand is first widened to binary32, which is exact;
- a 32-bit integer operand of the result's signedness is unchanged. Of the other signedness it is clamped by
  ``__cgl_sat_utoi`` or ``__cgl_sat_itou`` when saturating, and otherwise cast, which keeps the bits (modular, as
  C99 6.3.1.3 requires for an unsigned result);
- 8- and 16-bit integers and booleans are first widened to 32 bits, which is exact. An unsigned one then fits either
  result type.

Other operands (binary64, 64-bit integers, unknown types, more than four lanes) keep the generic call and get a reason
in ``annotations["cgl_rule_unresolved"]``; lowering refuses the call with that reason.
"""

from __future__ import annotations

from .hub_types import INT32, UINT32, HubType, call_name
from .rewrite import call
from .rule_table import CONVERSION_SIGNATURES, CONVERT_SAT_GENERIC

UNRESOLVED = "cgl_rule_unresolved"
_ROUNDING = {"rte": "roundEven", "rtp": "ceil", "rtn": "floor"}
_RESOLVED = {
    ("int", "f"): "__cgl_sat_ftoi",
    ("uint", "f"): "__cgl_sat_ftou",
    ("int", "u"): "__cgl_sat_utoi",
    ("uint", "i"): "__cgl_sat_itou",
}
_RESULT_TYPES = {"int": INT32, "uint": UINT32}
_SCALAR_NAMES = {"f": "float", "i": "int", "u": "uint"}
_VECTOR_PREFIXES = {"f": "vec", "i": "ivec", "u": "uvec"}
_KIND_NAMES = {
    "f": "float",
    "i": "signed integer",
    "u": "unsigned integer",
    "b": "bool",
}


def type_call(hub: HubType, operand):
    """``operand`` converted to ``hub``, a float, int or uint scalar or vector type."""
    if hub.lanes == 1:
        return call(_SCALAR_NAMES[hub.kind], [operand])
    return call(f"{_VECTOR_PREFIXES[hub.kind]}{hub.lanes}", [operand])


def native_conversion(intrinsic, operand, lanes):
    """The target's own conversion for a native entry: the cast to the intrinsic's result type."""
    result = _RESULT_TYPES[CONVERSION_SIGNATURES[intrinsic][1]]
    return type_call(result.with_lanes(lanes), operand)


def resolve(node, operand_type: HubType | None):
    """The replacement of the generic conversion call ``node`` for an operand of ``operand_type``, else ``node``."""
    result, saturate, mode = CONVERT_SAT_GENERIC[call_name(node)]
    arguments = list(node.arguments or [])
    if len(arguments) != 1:
        return _unresolved(node, f"takes one operand, got {len(arguments)}")
    if operand_type is None:
        return _unresolved(node, "has an operand whose type could not be inferred")
    if not 1 <= operand_type.lanes <= 4:
        return _unresolved(node, f"has no helper for {operand_type.lanes} lanes")
    (operand,) = arguments
    kind, bits, lanes = operand_type.kind, operand_type.bits, operand_type.lanes
    result_type = _RESULT_TYPES[result].with_lanes(lanes)
    if kind == "f" and bits in (16, 32):
        if bits == 16:
            operand = type_call(HubType("f", 32, lanes), operand)
        if mode in _ROUNDING:
            operand = call(_ROUNDING[mode], [operand])
        return _intrinsic(node, _RESOLVED[(result, "f")], operand, lanes)
    if kind in ("i", "u", "b") and bits <= 32:
        if kind == "b" or bits < 32:
            if kind != "i":
                # At most 16 value bits: the widened value fits int and uint alike.
                return _located(node, type_call(result_type, operand))
            operand = type_call(HubType("i", 32, lanes), operand)
        if kind == result_type.kind:
            return _located(node, operand)
        if not saturate:
            return _located(node, type_call(result_type, operand))
        return _intrinsic(node, _RESOLVED[(result, kind)], operand, lanes)
    return _unresolved(node, f"has no conversion from a {bits}-bit {_KIND_NAMES[kind]}")


def _intrinsic(node, intrinsic, operand, lanes):
    replacement = call(intrinsic, [operand])
    result = _RESULT_TYPES[CONVERSION_SIGNATURES[intrinsic][1]]
    replacement.annotations["cgl_rule_type"] = result.with_lanes(lanes)
    return _located(node, replacement)


def _located(node, replacement):
    location = getattr(node, "source_location", None)
    if location is not None and getattr(replacement, "source_location", None) is None:
        replacement.source_location = location
    return replacement


def _unresolved(node, reason):
    node.annotations[UNRESOLVED] = reason
    return node
