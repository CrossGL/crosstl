"""HLSL textures read with ``GatherRaw`` keep their declared texel type in CrossGL.

CrossGL's sampled texture types (``texture2D``, ``utexture2D``, ...) return four components, and the importer narrows
each read to the HLSL texture's declared texel type. Shader model 6.7's ``GatherRaw`` is defined only on a texture
declared with one 16-, 32- or 64-bit unsigned integer component (``Texture2D<uint>``): it returns the raw bits of
the four texels' elements. ``utexture2D`` is written back as ``Texture2D<uint4>``, which DXC rejects for
``GatherRaw``, so a texture that is a ``GatherRaw`` receiver keeps its HLSL spelling, ``Texture2D<uint>``, which the
HLSL generator declares as written.
"""

from __future__ import annotations

from .DirectxAst import FunctionCallNode, MemberAccessNode

__all__ = ["collect_raw_gather_texture_declaration_ids", "raw_gather_texture_type"]

RAW_GATHER_METHOD = "GatherRaw"
# The texture types HLSL defines GatherRaw on.
_RAW_GATHER_TEXTURES = frozenset(
    {"Texture2D", "Texture2DArray", "TextureCube", "TextureCubeArray"}
)


def _receiver_names(converter, root):
    names = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node is None or isinstance(node, (str, int, float, bool)):
            continue
        if (
            isinstance(node, FunctionCallNode)
            and isinstance(node.name, MemberAccessNode)
            and node.name.member == RAW_GATHER_METHOD
        ):
            name = converter.expression_base_name(node.name.object)
            if name:
                names.add(name)
        pending.extend(converter.iter_ast_children(node))
    return names


def collect_raw_gather_texture_declaration_ids(converter, root):
    """The ids of the global and parameter declarations of ``root`` that a ``GatherRaw`` call reads."""
    ids = set()
    global_receivers = set()
    for function in getattr(root, "functions", []) or []:
        receivers = _receiver_names(converter, getattr(function, "body", []))
        parameters = list(getattr(function, "params", []) or [])
        parameter_names = {getattr(parameter, "name", None) for parameter in parameters}
        for parameter in parameters:
            if getattr(parameter, "name", None) in receivers:
                ids.add(id(parameter))
        global_receivers.update(receivers - parameter_names)
    for node in getattr(root, "global_variables", []) or []:
        if getattr(node, "name", None) in global_receivers:
            ids.add(id(node))
    return ids


def raw_gather_texture_type(converter, hlsl_type):
    """The CrossGL spelling of a ``GatherRaw`` receiver's HLSL type, or ``None`` to map it as usual."""
    text = str(hlsl_type or "").strip()
    if "<" not in text or not text.endswith(">"):
        return None
    base, argument = text.split("<", 1)
    base = base.strip()
    if base not in _RAW_GATHER_TEXTURES:
        return None
    arguments = converter.split_generic_arguments(argument[:-1])
    if len(arguments) != 1:
        return None
    element = converter.map_type(arguments[0].strip())
    return f"{base}<{element}>"
