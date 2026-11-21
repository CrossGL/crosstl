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

  - ``quad_shuffle_xor(data, mask)`` reads lane ``(lane ^ mask) & 3``;
  - ``quad_shuffle_up(data, delta)`` reads lane ``lane - delta`` and leaves the lower ``delta`` lanes their own
    value, so it reads ``min(lane - delta, lane)`` in unsigned arithmetic, where ``lane - delta`` wraps above
    ``lane`` exactly when ``lane < delta``;
  - ``quad_shuffle_down(data, delta)`` is the same shift of the reversed lane order, ``3 - lane``, so it reads
    ``3 - min((3 - lane) - delta, 3 - lane)``;
  - ``quad_shuffle_rotate_up`` and ``quad_shuffle_rotate_down`` read ``(lane - delta) & 3`` and ``(lane + delta) & 3``.

  The mask or delta is evaluated once, before the read, as Metal evaluates it.

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
    "quad_shuffle_xor": "(({lane} ^ {operand}) & 3u)",
    "quad_shuffle_up": "min({lane} - {operand}, {lane})",
    "quad_shuffle_down": "(3u - min((3u - {lane}) - {operand}, 3u - {lane}))",
    "quad_shuffle_rotate_up": "(({lane} - {operand}) & 3u)",
    "quad_shuffle_rotate_down": "(({lane} + {operand}) & 3u)",
}
_INTEGER = re.compile(r"(0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*")


def _unqualified(name: Any) -> str:
    text = str(name).lstrip(":")
    return text[len("metal::") :] if text.startswith("metal::") else text


def _integer_literal(argument: Any) -> int | None:
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
        return f"QuadReadLaneAt({arguments[0]}, {operand})"
    if name == "quad_shuffle_xor" and literal is not None:
        across = _ACROSS.get(literal & 0xFFFF)
        if across is not None:
            return f"{across}({arguments[0]})"
    lane = pattern.format(lane=QUAD_LANE, operand=operand)
    return f"QuadReadLaneAt({arguments[0]}, {lane})"
