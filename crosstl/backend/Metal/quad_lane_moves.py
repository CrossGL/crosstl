"""CrossGL for the Metal quad-group functions whose CrossGL operation takes other arguments.

A Metal quad-group is four consecutive lanes of a SIMD-group (MSL 4.1, section 6.10.3), the quad of a CrossGL stage
that declares no derivative group. ``builtin_tables`` maps each quad-group function to its CrossGL operation; this
module writes the calls whose arguments differ:

* ``quad_broadcast`` and ``quad_shuffle`` are ``QuadReadLaneAt``; a constant lane stays a constant.
* ``quad_product``, ``quad_min``, ``quad_max``, ``quad_and``, ``quad_or`` and ``quad_xor`` reduce the active threads
  of the quad-group, which is the clustered reduction ``WaveClustered*(value, 4u)``.
* ``quad_shuffle_xor`` with the constant mask 1, 2 or 3 reads the horizontal, vertical or diagonal neighbour,
  ``QuadReadAcrossX``, ``QuadReadAcrossY`` or ``QuadReadAcrossDiagonal``.
* Every other permute reads the quad lane it computes from the caller's, ``WaveGetLaneIndex() & 3u``, with
  ``QuadReadLaneAt``, which accepts a lane that differs between threads:

  - ``quad_shuffle_xor(data, mask)`` reads lane ``lane ^ mask`` for a mask below 4;
  - ``quad_shuffle_up(data, delta)`` reads lane ``lane - delta`` and leaves the lower ``delta`` lanes their own
    value, so it reads ``min(lane - delta, lane)`` in unsigned arithmetic, where ``lane - delta`` wraps above
    ``lane`` exactly when ``lane < delta``;
  - ``quad_shuffle_down(data, delta)`` is the same shift of the reversed lane order, ``3 - lane``, so it reads
    ``3 - min((3 - lane) - delta, 3 - lane)``;
  - ``quad_shuffle_rotate_up`` and ``quad_shuffle_rotate_down`` read ``(lane - delta) & 3`` and ``(lane + delta) & 3``.

  The mask or delta is evaluated once, before the read, as Metal evaluates it.

* A lane, or a ``quad_shuffle_xor`` mask, of 4 or more names no lane of the quad-group, and Metal requires a valid
  one. Apple GPUs then return the caller's own value from ``quad_shuffle``, ``quad_broadcast`` and
  ``quad_shuffle_xor`` (measured on an M2 Max for operands up to 65535), so the import reads the caller's lane. A
  run-time operand ``n`` selects from the nibble table ``0x3210 | own << 16``, entry ``min(n, 4)``: lanes 0 to 3
  read themselves and anything larger the caller's lane, with ``n`` evaluated once. ``quad_shuffle_xor`` reads
  ``lane ^ m`` for the mask ``m`` that table ``0x3210`` gives, which is 0 above 3. The shifts and rotations of
  ``quad_shuffle_up``, ``quad_shuffle_down`` and ``quad_shuffle_rotate_*`` above already compute what Apple GPUs
  compute for every delta.

``quad_shuffle_and_fill_*``, the prefix scans, ``quad_broadcast_first``, ``quad_elect`` and ``quad_is_first`` have no
CrossGL operation yet; they keep their Metal names, which Metal output writes back and the other generators refuse.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["render_quad_lane_move"]

QUAD_LANE = "(WaveGetLaneIndex() & 3u)"
_CLUSTERED = frozenset(
    {"quad_product", "quad_min", "quad_max", "quad_and", "quad_or", "quad_xor"}
)
_ACROSS = {1: "QuadReadAcrossX", 2: "QuadReadAcrossY", 3: "QuadReadAcrossDiagonal"}
_LANES = {
    "quad_shuffle_xor": "({lane} ^ ((0x3210u >> (min({operand}, 4u) << 2u)) & 3u))",
    "quad_shuffle_up": "min({lane} - {operand}, {lane})",
    "quad_shuffle_down": "(3u - min((3u - {lane}) - {operand}, 3u - {lane}))",
    "quad_shuffle_rotate_up": "(({lane} - {operand}) & 3u)",
    "quad_shuffle_rotate_down": "(({lane} + {operand}) & 3u)",
}
# The lane ``quad_shuffle`` and ``quad_broadcast`` read for a run-time lane: the lane itself below 4, the caller's
# lane otherwise.
_NAMED_LANE = "(((0x3210u | ({lane} << 16u)) >> (min({operand}, 4u) << 2u)) & 3u)"
_INTEGER = re.compile(r"(0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*")
# The value bits each integer scalar type keeps of a converted constant.
_INTEGER_MASKS = {
    **dict.fromkeys(("char", "uchar", "int8_t", "uint8_t"), 0xFF),
    **dict.fromkeys(("short", "ushort", "int16_t", "uint16_t"), 0xFFFF),
    **dict.fromkeys(("int", "uint", "int32_t", "uint32_t"), 0xFFFFFFFF),
    **dict.fromkeys(("long", "ulong", "int64_t", "uint64_t", "size_t"), (1 << 64) - 1),
}


def _converted_operand(argument: Any) -> tuple[str, Any] | None:
    """``(type, operand)`` of a conversion of one operand to an integer scalar type, ``T(x)`` or ``(T)x``."""
    args = getattr(argument, "args", None)
    type_name = getattr(argument, "vector_type", None)
    if type_name is not None and isinstance(args, list) and len(args) == 1:
        operand = args[0]
    elif hasattr(argument, "target_type") and hasattr(argument, "expression"):
        type_name, operand = argument.target_type, argument.expression
    else:
        return None
    type_name = _unqualified(getattr(type_name, "name", type_name))
    return (type_name, operand) if type_name in _INTEGER_MASKS else None


def _unqualified(name: Any) -> str:
    text = str(name).lstrip(":")
    return text[len("metal::") :] if text.startswith("metal::") else text


def _integer_literal(argument: Any) -> int | None:
    """The value of an integer literal, also one converted to an integer type, as ``ushort(2)`` or ``(uint)1``."""
    converted = _converted_operand(argument)
    if converted is not None:
        type_name, operand = converted
        value = _integer_literal(operand)
        return None if value is None else value & _INTEGER_MASKS[type_name]
    if isinstance(argument, bool) or not isinstance(argument, (str, int)):
        return None
    if isinstance(argument, int):
        return argument
    match = _INTEGER.fullmatch(argument.strip())
    return None if match is None else int(match.group(1), 0)


def render_quad_lane_move(
    converter: Any,
    source_name: Any,
    function_name: str,
    expr: Any,
    arguments: list[str],
) -> str | None:
    """The CrossGL call for a Metal quad-group reduction or permute, or None when the call is no such function.

    ``function_name`` is the CrossGL name the call resolved to and ``arguments`` its generated arguments; a call
    that binds to a user function of the same name resolves to another name and is left alone.
    """
    name = _unqualified(source_name)
    if converter.metal_wave_intrinsics.get(name) != function_name:
        return None
    if name in _CLUSTERED:
        if len(arguments) != 1:
            return None
        return f"{function_name}({arguments[0]}, 4u)"
    if function_name != "QuadReadLaneAt" or len(arguments) != 2:
        return None
    literal = _integer_literal(expr.args[1])
    # The lane, mask or delta converts to ushort; a constant stays a constant, which a generator may need it to be.
    operand = arguments[1] if literal is None else f"{literal & 0xFFFF}u"
    pattern = _LANES.get(name)
    if pattern is None:
        if literal is None:
            lane = _NAMED_LANE.format(lane=QUAD_LANE, operand=operand)
        else:
            lane = operand if literal & 0xFFFF < 4 else QUAD_LANE
        return f"QuadReadLaneAt({arguments[0]}, {lane})"
    if name == "quad_shuffle_xor" and literal is not None:
        across = _ACROSS.get(literal & 0xFFFF)
        if across is not None:
            return f"{across}({arguments[0]})"
        # A mask of 0, or of 4 or more, reads the caller's own lane.
        return f"QuadReadLaneAt({arguments[0]}, {QUAD_LANE})"
    lane = pattern.format(lane=QUAD_LANE, operand=operand)
    return f"QuadReadLaneAt({arguments[0]}, {lane})"
