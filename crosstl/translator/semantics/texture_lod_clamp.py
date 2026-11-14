"""Minimum-LOD clamped texture sampling, refused for targets whose emitters cannot attach the clamp to a sample.

CrossGL's ``textureClamp`` family (:mod:`crosstl.translator.codegen.texture_lod_clamp`) is lowered by the HLSL,
MSL, desktop GLSL and SPIR-V emitters. The other targets used to print the CrossGL call verbatim, which no target
compiler accepts. :func:`refuse_texture_lod_clamp` refuses them before generation with
:class:`~crosstl.translator.codegen.texture_lod_clamp.TextureLodClampUnsupportedError`:

* GLSL ES 3.00 (WebGL 2) has no clamped sampling functions; GL_ARB_sparse_texture_clamp is desktop OpenGL only.
* WGSL's sampling built-ins take no minimum-LOD operand (WGSL section 17.7); WebGPU sets the clamp as sampler state
  (``GPUSamplerDescriptor.lodMinClamp``), shared by every sample through that sampler.
* CUDA and HIP texture fetches take no minimum-LOD operand either; the clamp is texture object state
  (``minMipmapLevelClamp`` of the texture descriptor).
* The Slang, Rust and Mojo emitters have no lowering for it.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.texture_lod_clamp import (
    TEXTURE_LOD_CLAMP_INTRINSIC_NAMES,
    TextureLodClampUnsupportedError,
)
from .hub_types import call_name
from .rewrite import child_slots, get_child

__all__ = ["LOD_CLAMP_TARGETS", "refuse_texture_lod_clamp"]

# Targets whose emitters lower the clamp (GLSL ES profiles of "opengl" are refused by its version requirements).
LOD_CLAMP_TARGETS = frozenset(
    {"directx", "hlsl", "metal", "opengl", "glsl", "vulkan", "spirv", "cgl", "crossgl"}
)
_REASONS = {
    "webgl": (
        "GLSL ES 3.00 has no clamped sampling functions, and GL_ARB_sparse_texture_clamp is a desktop OpenGL "
        "extension"
    ),
    "wgsl": (
        "WGSL sampling built-ins take no minimum-LOD operand; WebGPU sets the clamp as sampler state "
        "(GPUSamplerDescriptor.lodMinClamp), shared by every sample through the sampler"
    ),
    "cuda": (
        "CUDA texture fetches take no minimum-LOD operand; the clamp is texture object state "
        "(cudaTextureDesc.minMipmapLevelClamp)"
    ),
    "hip": (
        "HIP texture fetches take no minimum-LOD operand; the clamp is texture object state "
        "(hipTextureDesc.minMipmapLevelClamp)"
    ),
}


def _calls(program):
    stack = [program]
    while stack:
        node = stack.pop()
        if isinstance(node, A.FunctionCallNode):
            yield node
        children = [get_child(node, field, key) for field, key in child_slots(node)]
        stack.extend(reversed(children))


def refuse_texture_lod_clamp(program, target) -> None:
    """Raise for the first clamped texture operation in ``program`` when ``target`` cannot lower it."""
    target = str(target or "").lower()
    if target in LOD_CLAMP_TARGETS or not isinstance(program, A.ShaderNode):
        return
    for node in _calls(program):
        operation = call_name(node)
        if operation in TEXTURE_LOD_CLAMP_INTRINSIC_NAMES:
            raise TextureLodClampUnsupportedError(
                target,
                operation,
                _REASONS.get(
                    target,
                    f"the {target} emitter has no minimum-LOD operand for texture sampling",
                ),
                source_location=getattr(node, "source_location", None),
            )
