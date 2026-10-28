"""Typed refusal for constructs the HIP emitter cannot lower.

The HIP emitter generates through :meth:`~.lowering_refusals.LoweringRefusals.run`: a site that meets a construct
it cannot express either raises :class:`HipLoweringRefused` at once or leaves a ``/* unsupported ... */`` loss marker,
and a rendered program that carries a marker is refused with every marked reason. No program is returned with one.
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
