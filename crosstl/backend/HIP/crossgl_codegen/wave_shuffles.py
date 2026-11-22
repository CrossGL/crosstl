"""HIP's mask-less wave shuffles with a run-time width, and ``__ballot``, as CrossGL wave operations.

HIP's ``__shfl``, ``__shfl_up``, ``__shfl_down`` and ``__shfl_xor`` take an optional ``width``, a power of two up to
``warpSize``, and work inside the caller's ``width``-lane segment of the wavefront (HIP Programming Guide, "Warp
cross-lane functions"). A literal width imports through :meth:`hip_warp_shuffle_segment_width`; a width the program
computes at run time imports here with the lane arithmetic of HIP's own definitions (``amd_warp_functions.h``), which
for a power of two equals the literal-width forms:

- ``__shfl(v, src, w)`` reads lane ``(src & (w - 1)) + (lane & ~(w - 1))``;
- ``__shfl_up(v, d, w)`` reads lane ``lane - d``, or the caller's own lane when that is below its segment's first;
- ``__shfl_down(v, d, w)`` reads lane ``lane + d``, or the caller's own when ``(lane & (w - 1)) + d >= w``;
- ``__shfl_xor(v, m, w)`` reads lane ``lane ^ m``, or the caller's own when that is at or past ``(lane + w) & ~(w - 1)``.

``__ballot(p)`` returns a 64-bit mask with bit ``n`` set when wavefront lane ``n`` is active and ``p`` is non-zero.
CrossGL's ``WaveActiveBallot`` holds lane ``n`` as bit ``n % 32`` of component ``n / 32``, so the mask is its ``x``
component joined with its ``y`` component shifted up by 32 bits.

Each form names an operand more than once, so an operand with a side effect (a call, an assignment, ``++`` or
``--``) is refused rather than evaluated twice.
"""

from __future__ import annotations

import re

__all__ = ["ballot_mask", "is_repeatable", "is_runtime_width", "runtime_width_shuffle"]

_LANE = "WaveGetLaneIndex()"
# An integer literal width the importer did not accept as a segment width is not a power of two up to 64, which HIP
# leaves undefined; it stays refused instead of being read as a run-time width.
_INTEGER_LITERAL = re.compile(r"\(*\s*(?:0[xX][0-9a-fA-F]+|\d+)[uUlL]*\s*\)*")
# Names, literals, member and element accesses, operators and scalar conversions: an operand that reads the same
# value each time it is evaluated.
_REPEATABLE = re.compile(r"[\w\s.\[\]()+\-*/%&|^~<>!=?:]*")
_SIDE_EFFECTS = re.compile(r"\+\+|--|(?<![=!<>])=(?!=)")
_CALLED = re.compile(r"([A-Za-z_]\w*)\s*\(")
_CONVERSIONS = frozenset(
    "bool int uint int8 uint8 int16 uint16 int64 uint64 half float double".split()
)


def is_repeatable(expression: str) -> bool:
    """Whether evaluating the CrossGL ``expression`` twice reads what evaluating it once does."""
    text = str(expression)
    return (
        bool(_REPEATABLE.fullmatch(text))
        and not _SIDE_EFFECTS.search(text)
        and all(name in _CONVERSIONS for name in _CALLED.findall(text))
    )


def is_runtime_width(width: str) -> bool:
    """Whether the imported ``width`` argument is a value the program computes, not an integer literal."""
    return not _INTEGER_LITERAL.fullmatch(str(width))


def runtime_width_shuffle(function_name: str, value: str, lane: str, width: str) -> str:
    """The CrossGL read of HIP ``function_name(value, lane, width)`` for a ``width`` computed at run time."""
    w = f"uint({width})"
    low_bits = f"({w} - 1u)"
    segment_base = f"({_LANE} & ~{low_bits})"
    if function_name == "__shfl":
        return (
            f"WaveReadLaneAt({value}, ({segment_base} + (uint({lane}) & {low_bits})))"
        )
    if function_name == "__shfl_up":
        return (
            f"WaveReadLaneAt({value}, (((int({_LANE}) - int({lane})) < int({segment_base})) ? "
            f"{_LANE} : ({_LANE} - uint({lane}))))"
        )
    if function_name == "__shfl_down":
        return (
            f"WaveReadLaneAt({value}, (((({_LANE} & {low_bits}) + uint({lane})) >= {w}) ? "
            f"{_LANE} : ({_LANE} + uint({lane}))))"
        )
    partner = f"({_LANE} ^ uint({lane}))"
    segment_end = f"(({_LANE} + {w}) & ~{low_bits})"
    return (
        f"WaveReadLaneAt({value}, (({partner} >= {segment_end}) ? {_LANE} : {partner}))"
    )


def ballot_mask(predicate: str) -> str:
    """The CrossGL 64-bit mask HIP ``__ballot`` returns for the wave predicate ``predicate``."""
    ballot = f"WaveActiveBallot({predicate})"
    return f"(uint64({ballot}.x) | (uint64({ballot}.y) << 32))"
