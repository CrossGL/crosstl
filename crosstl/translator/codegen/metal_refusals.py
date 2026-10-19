"""Typed refusal for constructs the Metal emitter cannot lower.

The Metal emitter used to write ``/* unsupported Metal ...: reason */`` next to a zero value, an empty statement
or a stub helper and return the program. Every such site now records its reason through
:meth:`MetalCodeGen.refuse_lowering`, and generation raises :class:`MetalLoweringRefused` listing all of them.
The image-access helpers shared with the other emitters record into the same sink through
:func:`~.lowering_refusals.refuse_unsupported`.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX
from .lowering_refusals import LoweringRefusals, LoweringRefused

__all__ = ["MetalLoweringRefused", "metal_refusals"]


class MetalLoweringRefused(LoweringRefused):
    """One or more constructs in a program have no faithful Metal lowering."""

    target_name = "metal"
    label = "Metal"
    code = f"{CODE_PREFIX}metal-lowering-unsupported"


def metal_refusals():
    return LoweringRefusals(MetalLoweringRefused)
