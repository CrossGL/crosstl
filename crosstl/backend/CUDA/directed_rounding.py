"""CUDA and HIP intrinsics that round in a direction other than to nearest.

``__fadd_rd``, ``__fmul_ru``, ``__fsqrt_rz``, ``__fmaf_ieee_rd``, ``__float2half_ru``, ``__int2float_rz`` and their
siblings (CUDA Math API, Single and Double Precision Intrinsics and Type Casting Intrinsics; HIP declares the same
names) compute the IEEE 754 result rounded down (``_rd``), up (``_ru``) or toward zero (``_rz``). CrossGL, like the
languages it targets, has only the operations that round to nearest, so the importers write a call of one with a
``float`` or ``double`` result as a call of a CrossGL function that computes the directed result exactly in integer
arithmetic (:mod:`.directed_rounding_helpers`), and add the functions the program needs at its top.

The conversions to ``half`` and ``bfloat16`` (``__float2half_rd``, ``__double2bfloat16_ru``, ``__ushort2half_rz``
and so on) have no such function yet and are refused with :class:`DirectedRoundingUnsupportedError`. The
round-to-nearest variants (``_rn``) are the plain operations.
"""

from __future__ import annotations

import re

from ...translator.diagnostics import CODE_PREFIX, TranslationDiagnostic
from .directed_rounding_helpers import DirectedRoundingLibrary

__all__ = [
    "DirectedRoundingLibrary",
    "DirectedRoundingUnsupportedError",
    "directed_rounding_call",
    "insert_directed_rounding_functions",
    "refuse_directed_rounding",
]

DIRECTED_ROUNDING_INTRINSIC = re.compile(
    r"^__(?:"
    r"[fd](?:add|sub|mul|div|rcp|sqrt)"
    r"|fmaf(?:_ieee)?|fma"
    r"|float2half|float2bfloat16|double2float|double2half|double2bfloat16"
    r"|(?:u?int|u?ll|u?short)2(?:float|double|half|bfloat16)"
    r")_r[duz]$"
)
_DIRECTIONS = {"d": "down", "u": "up", "z": "toward zero"}


class DirectedRoundingUnsupportedError(TranslationDiagnostic):
    """A CUDA or HIP intrinsic that rounds down, up or toward zero to a type CrossTL cannot round to exactly."""

    code = f"{CODE_PREFIX}directed-rounding-unsupported"
    missing_capabilities = ("crossgl.directed-rounding",)


def refuse_directed_rounding(function_name) -> None:
    """Refuse a call of a directed-rounding intrinsic; other names pass."""
    name = str(function_name or "").lstrip(":")
    if DIRECTED_ROUNDING_INTRINSIC.match(name):
        raise DirectedRoundingUnsupportedError(
            f"'{name}' rounds its result {_DIRECTIONS[name[-1]]}; CrossTL computes the directed-rounding "
            "intrinsics exactly only for float and double results"
        )


def directed_rounding_call(library, function_name, args):
    """The CrossGL call that computes ``function_name(args)`` exactly, or None when another rule handles the name."""
    name = str(function_name or "").lstrip(":")
    if library is None or not library.handles(name):
        return None
    if len(args) != library.arity(name):
        raise DirectedRoundingUnsupportedError(
            f"'{name}' takes {library.arity(name)} arguments, and the call passes {len(args)}"
        )
    return library.call(name, args)


def insert_directed_rounding_functions(output, library, header):
    """Insert the functions ``library`` collected into ``output`` lines, after the ``header`` line."""
    if library is None or not library.wrappers:
        return
    position = output.index(header) + 1 if header in output else 0
    output[position:position] = [library.render(), ""]
