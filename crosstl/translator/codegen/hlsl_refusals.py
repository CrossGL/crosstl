"""Typed refusal for constructs the DirectX HLSL emitter cannot lower.

The HLSL emitter used to write ``/* unsupported HLSL ...: reason */`` or ``/* unsupported DirectX ...: reason */``
next to a zero value or in place of a dropped declaration and return the program. Every such site now records its
reason through :meth:`~.lowering_refusals.LoweringRefusalMixin.refuse_lowering`, the shared image-access helpers
record through :func:`~.lowering_refusals.refuse_unsupported`, and generation raises
:class:`DirectXLoweringRefused` listing all of them.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .lowering_refusals import LoweringRefused

__all__ = ["DirectXBufferLayoutError", "DirectXLoweringRefused"]


class DirectXLoweringRefused(LoweringRefused):
    """One or more constructs in a program have no faithful DirectX HLSL lowering."""

    target_name = "directx"
    label = "DirectX"
    aliases = ("HLSL",)
    code = f"{CODE_PREFIX}directx-lowering-unsupported"


class DirectXBufferLayoutError(TranslationDiagnostic):
    """A buffer block whose source layout HLSL declarations cannot reproduce."""

    code = f"{CODE_PREFIX}directx-buffer-layout-unsupported"

    def __init__(self, block, layout, reason):
        self.block = block
        self.layout = layout
        self.missing_capabilities = (f"directx.buffer-layout.{layout}",)
        super().__init__(
            f"Cannot reproduce the {layout} layout of block '{block}' in HLSL: {reason}",
            target="directx",
        )
