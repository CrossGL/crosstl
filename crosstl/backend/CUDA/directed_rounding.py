"""CUDA and HIP intrinsics that round in a direction other than to nearest.

``__fadd_rd``, ``__fmul_ru``, ``__fsqrt_rz``, ``__fmaf_ieee_rd``, ``__float2half_ru``, ``__int2float_rz`` and their
siblings (CUDA Math API, Single and Double Precision Intrinsics and Type Casting Intrinsics; HIP declares the same
names) compute the IEEE 754 result rounded down (``_rd``), up (``_ru``) or toward zero (``_rz``). The importers used
to write them as the plain operation, which rounds to nearest: ``__fadd_rd(1.0f, 1e-8f)`` became ``1.0 + 1e-8``, one
ULP above the source's result, with no diagnostic. CrossGL has no per-operation rounding mode and no other target
language offers one, so they are refused with :class:`DirectedRoundingUnsupportedError`. The round-to-nearest
variants (``_rn``) are the plain operations and stay supported.
"""

from __future__ import annotations

import re

from ...translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["DirectedRoundingUnsupportedError", "refuse_directed_rounding"]

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
    """A CUDA or HIP intrinsic that rounds down, up or toward zero, which CrossGL cannot express."""

    code = f"{CODE_PREFIX}directed-rounding-unsupported"
    missing_capabilities = ("crossgl.directed-rounding",)


def refuse_directed_rounding(function_name) -> None:
    """Refuse a call of a directed-rounding intrinsic; other names pass."""
    name = str(function_name or "").lstrip(":")
    if DIRECTED_ROUNDING_INTRINSIC.match(name):
        raise DirectedRoundingUnsupportedError(
            f"'{name}' rounds its result {_DIRECTIONS[name[-1]]}, and CrossGL has no operation with a rounding "
            "mode other than to nearest"
        )
