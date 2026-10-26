"""Shader Model 6.9 hit objects, refused for the targets whose generators cannot declare them.

CrossGL spells the hit object of HLSL Shader Model 6.9 (DirectX Raytracing 1.2, "Shader Execution Reordering") the
way HLSL does, ``dx::HitObject``, and the DirectX generator writes the type and its static methods
(``dx::HitObject::MakeNop()``) back unchanged. No other generator lowers them: each copied the HLSL spelling into
the target verbatim, which no other target compiler accepts. :func:`refuse_hit_objects` refuses those targets with
:class:`HitObjectUnsupportedError` before generation.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name

__all__ = ["HIT_OBJECT_TARGETS", "HitObjectUnsupportedError", "refuse_hit_objects"]

HIT_OBJECT_TYPE = "dx::HitObject"
# Targets that keep the HLSL spelling: the Direct3D generators and CrossGL itself.
HIT_OBJECT_TARGETS = frozenset(
    {"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12", "cgl", "crossgl"}
)


class HitObjectUnsupportedError(TranslationDiagnostic):
    """A Shader Model 6.9 hit object for a target whose generator cannot declare one."""

    def __init__(self, target, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.hit-object",)
        super().__init__(
            f"{HIT_OBJECT_TYPE} (HLSL Shader Model 6.9 shader execution reordering) cannot be carried to {target}: "
            "only the DirectX generator declares hit objects",
            target=target,
            code=f"{CODE_PREFIX}{target}-hit-object-unsupported",
        )


def _names_hit_object(node):
    if isinstance(node, A.NamedType):
        return node.name == HIT_OBJECT_TYPE
    if isinstance(node, A.FunctionCallNode):
        return call_name(node).startswith(f"{HIT_OBJECT_TYPE}::")
    return False


def refuse_hit_objects(program, target) -> None:
    """Raise :class:`HitObjectUnsupportedError` for the first hit object ``target`` cannot declare."""
    target = str(target or "").lower()
    if target in HIT_OBJECT_TARGETS or not isinstance(program, A.ShaderNode):
        return
    for node in program.walk():
        if _names_hit_object(node):
            raise HitObjectUnsupportedError(
                target, source_location=getattr(node, "source_location", None)
            )
