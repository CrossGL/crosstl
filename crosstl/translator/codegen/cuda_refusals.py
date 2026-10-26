"""Typed refusal for constructs the CUDA emitter cannot lower.

The CUDA emitter does not collect refusals through :class:`~.lowering_refusals.LoweringRefusals` yet, so it raises
:class:`CudaLoweringRefused` where it meets the construct.
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
