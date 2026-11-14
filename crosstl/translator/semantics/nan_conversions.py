"""Float-to-integer conversions of the ``ftoi_nan`` rule family: a NaN operand converts to 0.

MSL 4.1 section 8.6 defines converting a floating-point value to an integer type: the value rounds toward zero, and a
NaN gives 0. A truncation the integer type cannot represent stays undefined (section 2.23 converts with no
saturation, and C++17 [conv.fpint]/1 applies). GLSL, SPIR-V, DXC, CUDA and the other targets leave the NaN case
undefined as well, so without the rule a Metal conversion would lose its NaN result there.

:func:`carry_conversion` rewrites a conversion of a ``float`` or ``half`` scalar or vector to an 8-, 16- or 32-bit
integer type into ``__cgl_nan0_ftoi(x)``, or ``__cgl_nan0_ftou(x)`` for a 32-bit unsigned destination. That covers
explicit conversions (``T(x)``, ``(T)x``) and the implicit ones of an initializer or a plain assignment, which become
explicit. The call keeps the conversion in ``annotations["cgl_rule_origin"]``, its destination type in
``cgl_rule_type`` and its operand type in ``cgl_rule_operand``. Lowering (:func:`lower_conversion`) either restores
the conversion with its lowered operand (native) or calls a saturating ``convert_sat`` helper (polyfill). The helpers
map NaN to 0 and clamp what the rule leaves undefined, so they refine it:

- a binary16 operand is first widened to binary32, which is exact;
- a 32-bit destination takes the helper's result, splat by the conversion when the operand is a scalar;
- an 8- or 16-bit destination takes the helper's ``int`` clamped to the type's range, so NaN gives 0, a representable
  truncation is exact and other values saturate, as Apple GPUs do.

Conversions in call arguments, return values and buffer stores that stay implicit are not carried yet, and neither are
binary64 operands, which Metal does not have.
"""

from __future__ import annotations

from .. import ast as A
from .conversions import type_call
from .hub_types import FLOAT32, HubType, call_name, type_from_name, type_from_node
from .narrow import narrow_from_name, narrow_of
from .rewrite import call, set_child
from .rule_table import FTOI_NAN

ORIGIN = "cgl_rule_origin"
OPERAND = "cgl_rule_operand"
_SIGNED, _UNSIGNED = "__cgl_nan0_ftoi", "__cgl_nan0_ftou"
_SCALAR_NAMES = {
    ("i", 8): "int8",
    ("u", 8): "uint8",
    ("i", 16): "int16",
    ("u", 16): "uint16",
    ("i", 32): "int",
    ("u", 32): "uint",
}
_VECTOR_PREFIXES = {
    ("i", 8): "i8vec",
    ("u", 8): "u8vec",
    ("i", 16): "i16vec",
    ("u", 16): "u16vec",
    ("i", 32): "ivec",
    ("u", 32): "uvec",
}


def destination_type(type_value) -> HubType | None:
    """The integer value type a conversion to ``type_value`` produces, if the rule covers it, else ``None``."""
    if type_value is None:
        return None
    if isinstance(type_value, HubType):
        hub = type_value
    elif isinstance(type_value, str):
        if type_value.startswith("min"):
            # min16int and the like are minimum-precision types, not narrow ones.
            return None
        hub = type_from_name(type_value) or narrow_from_name(type_value)
    else:
        if str(getattr(type_value, "name", "") or "").startswith("min"):
            return None
        hub = type_from_node(type_value) or narrow_of(type_value)
    if hub is None or hub.kind not in ("i", "u") or hub.bits not in (8, 16, 32):
        return None
    return hub if 1 <= hub.lanes <= 4 else None


def type_name(hub: HubType) -> str:
    """The CrossGL spelling of an integer value type that :func:`destination_type` accepts."""
    key = (hub.kind, hub.bits)
    return (
        _SCALAR_NAMES[key] if hub.lanes == 1 else f"{_VECTOR_PREFIXES[key]}{hub.lanes}"
    )


def explicit_conversion(node):
    """``(operand, destination)`` of ``T(x)`` or ``(T)x`` to an integer type, else ``None``."""
    if isinstance(node, A.CastNode):
        destination = destination_type(node.target_type)
        return (node.expression, destination) if destination is not None else None
    if not isinstance(node, A.FunctionCallNode):
        return None
    arguments = node.arguments or []
    destination = destination_type(call_name(node))
    if destination is None or len(arguments) != 1:
        return None
    return arguments[0], destination


def carry_conversion(scope, origin, operand, destination, report):
    """``origin`` (converting ``operand`` to ``destination``) as an ``ftoi_nan`` intrinsic when the operand is floating.

    ``report.unresolved`` counts conversions whose operand type cannot be inferred.
    """
    source = scope.infer(operand)
    if source is None:
        report.unresolved += 1
        return origin
    if source.kind != "f" or source.bits not in (16, 32):
        return origin
    if source.lanes not in (1, destination.lanes):
        return origin
    report.count(FTOI_NAN)
    unsigned = destination.kind == "u" and destination.bits == 32
    replacement = A.FunctionCallNode(
        A.IdentifierNode(_UNSIGNED if unsigned else _SIGNED), [operand]
    )
    replacement.annotations[ORIGIN] = origin
    replacement.annotations["cgl_rule_type"] = destination
    replacement.annotations[OPERAND] = source
    return replacement


def implicit_conversion(scope, value, destination, report):
    """``value`` stored to a ``destination`` integer variable, with its conversion carried if it is floating."""
    origin = call(type_name(destination), [value])
    carried = carry_conversion(scope, origin, value, destination, report)
    return value if carried is origin else carried


def _set_operand(origin, operand):
    if isinstance(origin, A.CastNode):
        set_child(origin, "expression", None, operand)
    else:
        set_child(origin, "arguments", 0, operand)


def _bound(value, lanes):
    literal = A.LiteralNode(abs(value), A.PrimitiveType("int"))
    scalar = literal if value >= 0 else A.UnaryOpNode("-", literal)
    return scalar if lanes == 1 else call(f"ivec{lanes}", [scalar])


def lower_conversion(node, entry_kind, helpers):
    """The native or polyfill lowering of a carried conversion ``node`` (see the module docstring)."""
    origin = node.annotations[ORIGIN]
    destination = node.annotations["cgl_rule_type"]
    source = node.annotations[OPERAND]
    (operand,) = node.arguments
    if entry_kind == "native":
        _set_operand(origin, operand)
        return origin
    if source.bits == 16:
        operand = type_call(FLOAT32.with_lanes(source.lanes), operand)
    value = helpers.call(call_name(node), source.lanes, [operand])
    if destination.bits == 32 and destination.lanes == source.lanes:
        return value
    if destination.bits < 32:
        if destination.kind == "i":
            low, high = (
                -(1 << (destination.bits - 1)),
                (1 << (destination.bits - 1)) - 1,
            )
        else:
            low, high = 0, (1 << destination.bits) - 1
        lanes = source.lanes
        value = call("clamp", [value, _bound(low, lanes), _bound(high, lanes)])
    _set_operand(origin, value)
    return origin
