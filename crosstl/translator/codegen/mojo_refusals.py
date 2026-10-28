"""Typed refusal for constructs the Mojo emitter cannot lower.

Shared helpers refuse through :func:`~.lowering_refusals.refuse_unsupported` with the target ``"Mojo"``; the
Mojo emitter does not collect refusals, so such a call raises :class:`MojoLoweringRefused` at once.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX
from .lowering_refusals import LoweringRefused

__all__ = ["MojoLoweringRefused"]


class MojoLoweringRefused(LoweringRefused):
    """One or more constructs in a program have no faithful Mojo lowering."""

    target_name = "mojo"
    label = "Mojo"
    code = f"{CODE_PREFIX}mojo-lowering-unsupported"
