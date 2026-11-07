"""Storage-image atomics, which WGSL output refuses.

CrossGL performs an atomic read-modify-write on a storage-image texel with ``imageAtomicAdd``, ``imageAtomicMin``,
``imageAtomicMax``, ``imageAtomicAnd``, ``imageAtomicOr``, ``imageAtomicXor``, ``imageAtomicExchange`` and
``imageAtomicCompSwap``, as GLSL does (GLSL 4.60 section 8.12). WGSL has no atomic texture built-in: its texture
built-ins (WGSL section 17.7) read, write and query storage textures, and its atomic built-ins (section 17.8) take
a pointer to an ``atomic<T>`` in the ``storage`` or ``workgroup`` address space, which a texel is not. The emitter
used to print the call unchanged, which naga rejects as an unknown identifier, so it refuses these calls with
:class:`WGSLImageAtomicUnsupportedError` instead.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["WGSLImageAtomicUnsupportedError", "is_image_atomic_name"]


class WGSLImageAtomicUnsupportedError(TranslationDiagnostic):
    """An atomic operation on a storage-image texel, which no WGSL built-in performs."""

    def __init__(self, function_name, image):
        self.function_name = function_name
        self.image = image
        self.missing_capabilities = ("wgsl.storage-texture-atomics",)
        super().__init__(
            f"WGSL target cannot lower {function_name}() on {image}: WGSL has no atomic operation on a "
            "storage-texture texel; keep the values in a storage buffer of atomic<u32> or atomic<i32> instead",
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-image-atomic-unsupported",
        )


def is_image_atomic_name(normalized_name) -> bool:
    """Whether ``normalized_name``, a lower-case call name, is one of CrossGL's storage-image atomics."""
    return isinstance(normalized_name, str) and normalized_name.startswith(
        "imageatomic"
    )
