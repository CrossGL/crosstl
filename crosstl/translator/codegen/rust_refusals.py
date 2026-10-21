"""Typed refusal for constructs the Rust emitter cannot lower.

The Rust emitter used to write ``/* unsupported ...: reason */`` in place of a synchronization call, a subgroup
operation or a GLSL buffer-block access, next to a default value or in place of a dropped store, and return the
program. Every such site now records its reason through
:meth:`~.lowering_refusals.LoweringRefusalMixin.refuse_lowering`, and generation raises
:class:`RustLoweringRefused` listing all of them once the program is rendered.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX
from .lowering_refusals import LoweringRefused

__all__ = ["RustLoweringRefused"]


class RustLoweringRefused(LoweringRefused):
    """One or more constructs in a program have no faithful Rust lowering."""

    target_name = "rust"
    label = "Rust"
    code = f"{CODE_PREFIX}rust-lowering-unsupported"
