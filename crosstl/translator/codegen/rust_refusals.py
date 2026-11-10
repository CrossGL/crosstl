"""Typed refusal for constructs the Rust emitter cannot lower.

The Rust emitter used to write ``/* unsupported ...: reason */`` in place of a synchronization call, a subgroup
operation or a GLSL buffer-block access, next to a default value or in place of a dropped store, and return the
program. Every such site now records its reason through
:meth:`~.lowering_refusals.LoweringRefusalMixin.refuse_lowering`, and generation raises
:class:`RustLoweringRefused` listing all of them once the program is rendered.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .lowering_refusals import LoweringRefused

__all__ = ["RustEnumEmptyBraceError", "RustLoweringRefused"]


class RustLoweringRefused(LoweringRefused):
    """One or more constructs in a program have no faithful Rust lowering."""

    target_name = "rust"
    label = "Rust"
    code = f"{CODE_PREFIX}rust-lowering-unsupported"


class RustEnumEmptyBraceError(TranslationDiagnostic):
    """An empty brace initializer of an enumeration with no zero-valued plain variant."""

    def __init__(self, enum_name, detail):
        self.enum_name = enum_name
        self.missing_capabilities = ("rust.enum-empty-brace",)
        super().__init__(
            f"Rust target cannot write the empty brace initializer of enumeration "
            f"'{enum_name}': {detail}",
            target="rust",
            code=f"{CODE_PREFIX}rust-enum-empty-brace-unsupported",
        )
