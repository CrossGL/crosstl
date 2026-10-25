"""Typed refusal for constructs the HIP emitter cannot lower.

The HIP emitter does not collect refusals through :class:`~.lowering_refusals.LoweringRefusals` yet, so it raises
:class:`HipLoweringRefused` where it meets the construct.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX
from .lowering_refusals import LoweringRefused

__all__ = ["HipLoweringRefused"]


class HipLoweringRefused(LoweringRefused):
    """One or more constructs in a program have no faithful HIP lowering."""

    target_name = "hip"
    label = "HIP"
    code = f"{CODE_PREFIX}hip-lowering-unsupported"
