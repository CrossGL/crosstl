"""CrossGL pointer types that WGSL output cannot spell.

CrossGL spells an imported pointer local as ``ptr<T>``, for example
``var src: ptr<f32> = in_;`` for an OpenCL ``__global const float *src``.
WGSL spells a pointer ``ptr<address-space, T>``: the CrossGL spelling
carries no address space, and a local variable cannot be declared with a
pointer type at all. Such a type therefore has no WGSL spelling and is
refused with :class:`WGSLPointerTypeUnsupportedError` instead of being
copied unchanged, which naga rejects as an undeclared identifier.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["WGSLPointerTypeUnsupportedError"]


class WGSLPointerTypeUnsupportedError(TranslationDiagnostic):
    """A CrossGL ``ptr<T>`` type that WGSL output cannot spell."""

    def __init__(self, spelling):
        self.spelling = spelling
        self.missing_capabilities = ("wgsl.pointer-type",)
        super().__init__(
            f"WGSL target cannot represent CrossGL pointer type '{spelling}': "
            "the spelling carries no address space, and WGSL pointers must "
            "name one (ptr<address-space, T>)",
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-pointer-type-unsupported",
        )
