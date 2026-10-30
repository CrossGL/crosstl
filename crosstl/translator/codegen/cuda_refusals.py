"""Typed refusal for constructs the CUDA emitter cannot lower.

The CUDA emitter generates through :meth:`~.lowering_refusals.LoweringRefusals.run`: a site that meets a construct
it cannot express either raises :class:`CudaLoweringRefused` at once or leaves a ``/* unsupported ... */`` loss marker,
and a rendered program that carries a marker is refused with every marked reason. No program is returned with one.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX
from .lowering_refusals import LoweringRefused

__all__ = ["CudaLoweringRefused"]


class CudaLoweringRefused(LoweringRefused):
    """One or more constructs in a program have no faithful CUDA lowering."""

    target_name = "cuda"
    label = "CUDA"
    code = f"{CODE_PREFIX}cuda-lowering-unsupported"
