"""CUDA and HIP bfloat16 conversion intrinsics, imported as CrossGL ``bfloat16`` operations.

CUDA's ``cuda_bf16.h`` (CUDA Math API, "Bfloat16 Precision Conversion and Data Movement") and HIP's
``amd_hip_bf16.h`` declare the same names. ``__float2bfloat16`` and the ``_rn`` variants round a binary32 value to
the nearest bfloat16, ties to even, which is what a CrossGL ``bfloat16(x)`` conversion does; ``__bfloat162float``
widens exactly. The importers used to pass these names through unchanged, so the CrossGL program called an
undeclared function. Directed-rounding variants (``_rd``, ``_ru``, ``_rz``) are refused earlier by
``CUDA.directed_rounding``. Conversions from ``double`` or from integers are not listed: CrossGL would round twice
(to binary32, then to bfloat16), so they stay unrecognised and are refused downstream as undeclared functions.
"""

from __future__ import annotations

__all__ = ["BFLOAT16_CONVERSIONS", "format_bfloat16_conversion"]

_SCALAR = "bfloat16"


def _vector(elements, element_type=_SCALAR):
    return f"vec2<{element_type}>({', '.join(elements)})"


def _component(expression, component):
    text = str(expression).strip()
    if text and all(char.isalnum() or char in "_." for char in text):
        return f"{text}.{component}"
    return f"({text}).{component}"


def _to_bfloat16(args):
    return f"{_SCALAR}({args[0]})"


# name -> (arity, formatter)
BFLOAT16_CONVERSIONS = {
    "__float2bfloat16": (1, _to_bfloat16),
    "__float2bfloat16_rn": (1, _to_bfloat16),
    "__bfloat162float": (1, lambda args: f"f32({args[0]})"),
    # A one-argument vector constructor splats a scalar and converts a vector componentwise, so each
    # operand is evaluated once.
    "__float2bfloat162_rn": (1, lambda args: _vector([_to_bfloat16(args)])),
    "__floats2bfloat162_rn": (
        2,
        lambda args: _vector([f"{_SCALAR}({args[0]})", f"{_SCALAR}({args[1]})"]),
    ),
    "__float22bfloat162_rn": (1, lambda args: _vector([args[0]])),
    "__bfloat1622float2": (1, lambda args: _vector([args[0]], "f32")),
    "__bfloat162bfloat162": (1, lambda args: _vector([args[0]])),
    "__halves2bfloat162": (2, lambda args: _vector([args[0], args[1]])),
    "__low2bfloat16": (1, lambda args: _component(args[0], "x")),
    "__high2bfloat16": (1, lambda args: _component(args[0], "y")),
}


def format_bfloat16_conversion(function_name, args):
    """Return the CrossGL form of a bfloat16 conversion intrinsic, or ``None`` for any other call."""
    name = str(function_name or "").lstrip(":")
    entry = BFLOAT16_CONVERSIONS.get(name)
    if entry is None or len(args) != entry[0]:
        return None
    return entry[1](list(args))
