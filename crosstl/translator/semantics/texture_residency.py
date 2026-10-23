"""Texture residency queries, checked and refused for targets whose emitters do not lower them.

CrossGL's ``textureResidency(read, texel)`` and ``textureResident(code)``
(:mod:`crosstl.translator.codegen.texture_residency`) are lowered by the MSL and desktop GLSL emitters. Every other
target used to print the CrossGL call verbatim, which no target compiler accepts. :func:`refuse_texture_residency`
checks the shape of every query and refuses the others before generation with
:class:`~crosstl.translator.codegen.texture_residency.TextureResidencyUnsupportedError`:

* GLSL ES 3.00 (WebGL 2) has no sparse textures; GL_ARB_sparse_texture2 is a desktop OpenGL extension.
* WGSL has no sparse textures or residency feedback (WGSL section 17.7).
* CUDA and HIP texture fetches report no residency.
* The HLSL (``Status`` operand, ``CheckAccessFullyMapped``) and SPIR-V (``OpImageSparse*``) emitters, and the
  Slang, Rust and Mojo emitters, have no lowering for it yet.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.texture_residency import (
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENT_OPERATION,
    TextureResidencyUnsupportedError,
    split_texture_residency_call,
)
from .hub_types import call_name
from .rewrite import child_slots, get_child

__all__ = ["RESIDENCY_TARGETS", "refuse_texture_residency"]

# Targets whose emitters lower residency queries (GLSL ES profiles of "opengl" are refused by its version
# requirements).
RESIDENCY_TARGETS = frozenset({"metal", "opengl", "glsl", "cgl", "crossgl"})
_REASONS = {
    "webgl": (
        "GLSL ES 3.00 has no sparse textures, and GL_ARB_sparse_texture2 is a desktop OpenGL extension"
    ),
    "wgsl": "WGSL has no sparse textures or residency feedback (WGSL section 17.7)",
    "cuda": "CUDA texture fetches report no tile residency",
    "hip": "HIP texture fetches report no tile residency",
}


def _calls(program):
    stack = [program]
    while stack:
        node = stack.pop()
        if isinstance(node, A.FunctionCallNode):
            yield node
        children = [get_child(node, field, key) for field, key in child_slots(node)]
        stack.extend(reversed(children))


def refuse_texture_residency(program, target) -> None:
    """Raise for the first residency query in ``program`` that is malformed or that ``target`` cannot lower."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode):
        return
    queries = [
        node
        for node in _calls(program)
        if call_name(node) in (TEXTURE_RESIDENCY_OPERATION, TEXTURE_RESIDENT_OPERATION)
    ]
    # A malformed query is reported as such for every target, before any target refusal.
    for node in queries:
        if call_name(node) == TEXTURE_RESIDENCY_OPERATION:
            split_texture_residency_call(target or "cgl", node.arguments, node)
    if not queries or target in RESIDENCY_TARGETS:
        return
    node = queries[0]
    raise TextureResidencyUnsupportedError(
        target,
        call_name(node),
        _REASONS.get(
            target,
            f"the {target} emitter has no lowering for texture residency feedback",
        ),
        source_location=getattr(node, "source_location", None),
    )
