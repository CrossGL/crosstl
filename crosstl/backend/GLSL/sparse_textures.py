"""Import the GL_ARB_sparse_texture2 and GL_ARB_sparse_texture_clamp functions as CrossGL residency queries.

``int code = sparseTextureARB(s, P, texel[, bias])`` reads ``texture(s, P[, bias])`` into ``texel`` and returns its
residency code; ``sparseTexelsResidentARB(code)`` tests it. CrossGL wraps the ordinary read instead
(:mod:`crosstl.translator.codegen.texture_residency`)::

    int code = textureResidency(texture(s, P[, bias]), texel);
    bool resident = textureResident(code);

The read is imported through the ordinary call path, so shadow samplers become CrossGL's compare forms and Vulkan
GLSL ``samplerX(t, s)`` operands the ``t, s`` pair, exactly as for the non-sparse function. Passing the GLSL call
through as an ordinary call left it to the generators, which emitted it verbatim for every target.
"""

from __future__ import annotations

from crosstl.translator.codegen.texture_residency import (
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENT_OPERATION,
    TextureResidencyUnsupportedError,
)

from .OpenglAst import FunctionCallNode, VariableNode

__all__ = ["SPARSE_READS", "import_sparse_texture_call"]

# GLSL sparse function -> (the read it reports on, the operands before the ``out`` texel when an optional bias or
# gather component may follow it, or None when the texel is always last). GL_ARB_sparse_texture2, section 8.9.
SPARSE_READS = {
    "sparseTextureARB": ("texture", 2),
    "sparseTextureOffsetARB": ("textureOffset", 3),
    "sparseTextureLodARB": ("textureLod", None),
    "sparseTextureLodOffsetARB": ("textureLodOffset", None),
    "sparseTextureGradARB": ("textureGrad", None),
    "sparseTextureGradOffsetARB": ("textureGradOffset", None),
    "sparseTexelFetchARB": ("texelFetch", None),
    "sparseTexelFetchOffsetARB": ("texelFetchOffset", None),
    "sparseTextureGatherARB": ("textureGather", 2),
    "sparseTextureGatherOffsetARB": ("textureGatherOffset", 3),
    "sparseTextureGatherOffsetsARB": ("textureGatherOffsets", 3),
    "sparseImageLoadARB": ("imageLoad", None),
    "sparseTextureClampARB": ("textureClampARB", 3),
    "sparseTextureOffsetClampARB": ("textureOffsetClampARB", 4),
    "sparseTextureGradClampARB": ("textureGradClampARB", None),
    "sparseTextureGradOffsetClampARB": ("textureGradOffsetClampARB", None),
}
_GATHERS = frozenset({"textureGather", "textureGatherOffset", "textureGatherOffsets"})


def _texel_index(codegen, name, read, before_trailer, args):
    """The index of the ``out`` texel among ``args``, or None when the call fits no overload."""
    if before_trailer is None:
        return len(args) - 1 if len(args) >= 2 else None
    sampler_type = codegen.expression_shadow_sampler_type(args[0]) if args else None
    if sampler_type is not None:
        if read in _GATHERS:
            # A depth gather takes its reference before the texel, and no component (section 8.9.4).
            return before_trailer + 1 if len(args) == before_trailer + 2 else None
        if read == "texture" and sampler_type == "samplerCubeArrayShadow":
            # sparseTextureARB(samplerCubeArrayShadow, vec4 P, float compare, out float texel).
            return 3 if len(args) == 4 else None
    # The texel follows the mandatory operands, before an optional bias or gather component.
    return (
        before_trailer
        if len(args) in (before_trailer + 1, before_trailer + 2)
        else None
    )


def import_sparse_texture_call(codegen, name, args):
    """CrossGL for a GLSL sparse texture call, or None when ``name`` is not one."""
    args = list(args or [])
    if name == "sparseTexelsResidentARB":
        if len(args) != 1:
            raise TextureResidencyUnsupportedError(
                "cgl",
                name,
                f"sparseTexelsResidentARB takes one residency code, got {len(args)} operand(s)",
            )
        return f"{TEXTURE_RESIDENT_OPERATION}({codegen.generate_expression(args[0])})"
    entry = SPARSE_READS.get(name)
    if entry is None:
        return None
    read, before_trailer = entry
    index = _texel_index(codegen, name, read, before_trailer, args)
    if index is None:
        raise TextureResidencyUnsupportedError(
            "cgl",
            name,
            f"GL_ARB_sparse_texture2 has no {name} overload with {len(args)} operand(s) for this sampler",
        )
    texel = args[index]
    read_call = FunctionCallNode(
        VariableNode("", read), args[:index] + args[index + 1 :]
    )
    read_text = codegen.generate_function_call(read_call)
    return (
        f"{TEXTURE_RESIDENCY_OPERATION}({read_text}, "
        f"{codegen.generate_expression(texel)})"
    )
