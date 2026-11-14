"""System headers of Apple's SDKs that Metal sources include and no other target has.

A Metal shader includes ``<metal_stdlib>`` and its siblings, and the header it shares with the host app usually
includes ``<simd/simd.h>`` (the vector and matrix types both sides name, ``vector_float4``, ``matrix_float4x4``)
and ``<TargetConditionals.h>`` (the ``TARGET_OS_*`` macros). The CrossGL import reads the shared header's
declarations and maps the simd types to CrossGL's. The system headers themselves exist only in Apple's SDKs, so an
HLSL or GLSL translation that kept ``#include <simd/simd.h>`` fails with "file not found".
"""

from __future__ import annotations

__all__ = ["is_apple_sdk_include"]

_HEADERS = frozenset({"targetconditionals.h", "simd.h"})
_PREFIXES = ("metal_", "simd/")


def is_apple_sdk_include(name) -> bool:
    """Whether ``name``, an included path lower-cased with ``/`` separators, is an Apple SDK system header."""
    normalized = str(name or "").strip().strip("<>\"'").replace("\\", "/").lower()
    return normalized in _HEADERS or normalized.startswith(_PREFIXES)
