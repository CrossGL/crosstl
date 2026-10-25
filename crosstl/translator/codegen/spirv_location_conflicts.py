"""Typed refusal for two SPIR-V interface variables of one entry point at the same location.

The ``Input`` or ``Output`` variables an entry point lists must not share a location component (SPIR-V 1.6 section
2.16.2, Vulkan "Location Assignment"). A source that places two of its stage inputs or outputs at one location
cannot be expressed. The generator raised a plain ``ValueError`` for it, which callers could not tell apart from an
internal failure; :class:`InterfaceLocationConflictError` is a typed diagnostic that still matches ``ValueError``.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["InterfaceLocationConflictError"]


class InterfaceLocationConflictError(TranslationDiagnostic):
    """Two interface variables of one SPIR-V entry point take the same location."""

    code = CODE_PREFIX + "vulkan-interface-location-conflict"
    missing_capabilities = ("vulkan.aliased-interface-location",)

    def __init__(self, storage_class: str, location: int, name: str | None = None):
        self.storage_class = storage_class
        self.location = location
        self.variable_name = name
        named = f" for {name!r}" if name else ""
        super().__init__(
            f"Duplicate SPIR-V {storage_class.lower()} location {location}{named}: another "
            f"{storage_class.lower()} variable of the entry point already takes it, and Vulkan "
            "requires distinct locations within one interface",
            target="vulkan",
        )
