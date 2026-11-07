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

__all__ = [
    "DirectXBufferLayoutError",
    "DirectXLoweringRefused",
    "DirectXStageAttributesMissingError",
]


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


class DirectXStageAttributesMissingError(TranslationDiagnostic):
    """A geometry, hull or domain entry whose HLSL attributes the program does not give.

    HLSL fixes a hull shader's ``domain``, ``partitioning``, ``outputtopology``, ``outputcontrolpoints`` and
    ``patchconstantfunc``, a domain shader's ``domain`` and a geometry shader's ``maxvertexcount`` on the entry
    point. GLSL and SPIR-V spread the tessellation ones over the control and evaluation stages, so a stage translated
    on its own may not carry them.
    """

    code = f"{CODE_PREFIX}directx-stage-attributes-missing"

    def __init__(self, stage, missing):
        self.stage = stage
        self.missing = tuple(missing)
        self.missing_capabilities = tuple(
            f"directx.stage-attribute.{name}" for name in self.missing
        )
        super().__init__(
            f"DirectX {stage} stage requires HLSL attribute(s): {', '.join(self.missing)}; HLSL fixes them on "
            "the entry point, which carries them as @-attributes, and a GLSL or SPIR-V tessellation stage holds "
            "only part of them",
            target="directx",
        )
