"""Typed refusals for DirectX storage-pointer aliases that have no faithful HLSL lowering.

HLSL has no pointers, so the DirectX generator rewrites a storage pointer as its backing buffer plus an element
offset. When an alias, an assignment to it, or a store through it cannot keep the backing buffer's element type or
access, the generator refuses with :class:`DirectXResourcePointerAliasError` instead of emitting the access.
"""

from __future__ import annotations

__all__ = ["DirectXResourcePointerAliasError"]


class DirectXResourcePointerAliasError(ValueError):
    """A storage-pointer alias, assignment or dereference that HLSL cannot express."""

    project_diagnostic_code = (
        "project.translate.directx-resource-pointer-alias-unsupported"
    )
    missing_capabilities = ("directx.resource-pointer-alias-lowering",)

    def __init__(
        self,
        message,
        *,
        function_name=None,
        pointer_name=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.function_name = function_name
        self.pointer_name = pointer_name
        self.reason = reason
        self.source_location = source_location
