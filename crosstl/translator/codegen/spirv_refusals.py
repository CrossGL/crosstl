"""Typed refusal for SPIR-V lowerings the Vulkan emitter cannot perform.

The emitter used to write a ``; WARNING: ...`` comment into the module and carry on with a default value, an
omitted store or a dropped instruction. The comment is invisible to every SPIR-V consumer, so the module
assembled, validated and ran with different results. Each such site now records its reason here instead, and
:meth:`SpirvRefusals.raise_pending` stops the translation when the module is complete, reporting every reason at
once so that one run shows all the constructs that need work.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["SpirvLoweringRefused", "SpirvRefusals"]

CODE = f"{CODE_PREFIX}vulkan-lowering-unsupported"


class SpirvLoweringRefused(TranslationDiagnostic):
    """One or more constructs in a module have no faithful SPIR-V lowering.

    ``reasons`` lists every refusal in emission order, one entry per site; the message carries all of them.
    """

    def __init__(self, reasons):
        self.reasons = tuple(reasons)
        self.missing_capabilities = ("vulkan.lowering",)
        listed = "; ".join(self.reasons)
        count = len(self.reasons)
        noun = "construct" if count == 1 else "constructs"
        super().__init__(
            f"SPIR-V cannot lower {count} {noun}: {listed}", target="vulkan", code=CODE
        )


class SpirvRefusals:
    """The refusal reasons recorded while one module is generated."""

    def __init__(self):
        self.reasons = []

    def reset(self):
        self.reasons = []

    def record(self, reason):
        text = str(reason).strip()
        if text:
            self.reasons.append(text)

    def raise_pending(self, cause=None):
        if not self.reasons:
            return
        error = SpirvLoweringRefused(self.reasons)
        self.reasons = []
        if cause is not None:
            raise error from cause
        raise error
