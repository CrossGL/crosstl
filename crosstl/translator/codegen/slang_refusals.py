"""Typed refusal for constructs the Slang emitter cannot lower.

The Slang emitter used to write ``/* unsupported Slang ...: reason */`` next to a zero value, or in place of a dropped
store, and return the program. Every such site now records its reason through
:meth:`~.lowering_refusals.LoweringRefusalMixin.refuse_lowering`, and generation raises
:class:`SlangLoweringRefused` listing all of them once the program is rendered.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX
from .lowering_refusals import LoweringRefused

__all__ = ["SlangLoweringRefused"]


class SlangLoweringRefused(LoweringRefused):
    """One or more constructs in a program have no faithful Slang lowering."""

    target_name = "slang"
    label = "Slang"
    code = f"{CODE_PREFIX}slang-lowering-unsupported"
