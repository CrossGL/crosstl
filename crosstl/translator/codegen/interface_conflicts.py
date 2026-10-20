"""Typed refusal for stage interfaces whose locations or semantics overlap.

Two members of one stage interface that claim the same location (GLSL 4.60 section 4.4.2.1) or the same semantic
index (an HLSL array member ``float3 c[2] : TEXCOORD2`` takes ``TEXCOORD2`` and ``TEXCOORD3``) cannot both be
declared; the source is ill-formed for the target, which DXC's DXIL validator and glslang also reject.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["InterfaceLocationConflictError"]


class InterfaceLocationConflictError(TranslationDiagnostic):
    """Stage interface members whose locations or semantic indices overlap."""

    def __init__(self, target, detail):
        self.missing_capabilities = (f"{target}.interface-location-overlap",)
        super().__init__(
            detail,
            target=target,
            code=f"{CODE_PREFIX}{target}-interface-location-conflict",
        )
