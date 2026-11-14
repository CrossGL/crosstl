"""Texture residency queries, checked and refused for targets whose emitters do not lower them.

CrossGL's ``textureResidency(read, texel)`` and ``textureResident(code)``
(:mod:`crosstl.translator.codegen.texture_residency`) are lowered by the MSL, desktop GLSL, HLSL and SPIR-V emitters. Every other
target used to print the CrossGL call verbatim, which no target compiler accepts. :func:`refuse_texture_residency`
checks the shape of every query and refuses the others before generation with
:class:`~crosstl.translator.codegen.texture_residency.TextureResidencyUnsupportedError`:

* GLSL ES 3.00 (WebGL 2) has no sparse textures; GL_ARB_sparse_texture2 is a desktop OpenGL extension.
* WGSL has no sparse textures or residency feedback (WGSL section 17.7).
* CUDA and HIP texture fetches report no residency.
* The Slang, Rust and Mojo emitters have no lowering for it yet.

For HLSL targets, :func:`declare_hlsl_residency_status` declares one ``uint`` local per query at the top of its
function, to receive the read's ``out uint Status`` operand, and names it on the call
(:mod:`crosstl.translator.codegen.hlsl_texture_residency`).
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.texture_residency import (
    HLSL_RESIDENCY_STATUS_ATTRIBUTE,
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENT_OPERATION,
    TextureResidencyUnsupportedError,
    split_texture_residency_call,
)
from .hub_types import call_name, iter_functions
from .rewrite import child_slots, get_child

__all__ = [
    "HLSL_TARGETS",
    "RESIDENCY_TARGETS",
    "declare_hlsl_residency_status",
    "refuse_texture_residency",
]

# Targets whose emitters lower residency queries (GLSL ES profiles of "opengl" are refused by its version
# requirements).
HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
RESIDENCY_TARGETS = (
    frozenset({"metal", "opengl", "glsl", "vulkan", "spirv", "cgl", "crossgl"})
    | HLSL_TARGETS
)
_STATUS_PREFIX = "crosstl_residency_status_"
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


def _identifiers(program):
    names = set()
    stack = [program]
    while stack:
        node = stack.pop()
        for attribute in ("name", "identifier"):
            value = getattr(node, attribute, None)
            if isinstance(value, str):
                names.add(value)
        stack.extend(get_child(node, field, key) for field, key in child_slots(node))
    return names


def declare_hlsl_residency_status(program, target) -> int:
    """Declare a ``uint`` Status local per residency query at the top of its function, for HLSL targets.

    A declaration without an initializer evaluates nothing, so the top of the function is in scope of every query
    in it, a loop condition included. Returns the number of locals declared.
    """
    if (
        not isinstance(program, A.ShaderNode)
        or str(target or "").lower() not in HLSL_TARGETS
    ):
        return 0
    taken = None
    declared = 0
    for function in iter_functions(program):
        body = getattr(function, "body", None)
        statements = body.statements if isinstance(body, A.BlockNode) else body
        if not isinstance(statements, list):
            continue
        queries = [
            node
            for node in _calls(body)
            if call_name(node) == TEXTURE_RESIDENCY_OPERATION
            and not getattr(node, HLSL_RESIDENCY_STATUS_ATTRIBUTE, None)
        ]
        if not queries:
            continue
        if taken is None:
            taken = _identifiers(program)
        locals_ = []
        for node in queries:
            while f"{_STATUS_PREFIX}{declared}" in taken:
                declared += 1
            name = f"{_STATUS_PREFIX}{declared}"
            taken.add(name)
            declared += 1
            setattr(node, HLSL_RESIDENCY_STATUS_ATTRIBUTE, name)
            locals_.append(A.VariableNode(name, A.PrimitiveType("uint")))
        statements[:0] = locals_
    return declared
