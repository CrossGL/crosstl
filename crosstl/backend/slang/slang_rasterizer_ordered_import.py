"""CrossGL import of Slang's rasterizer-ordered views.

Slang has HLSL's ``RasterizerOrdered*`` resource types: read-write views whose
accesses from overlapping fragments happen in primitive order. They read and
write exactly like the matching ``RW*`` views. CrossGL writes them the way the
HLSL importer does, as the read-write resource with a ``@rasterizer_ordered``
qualifier: ``RasterizerOrderedTexture2D<float4> t`` becomes
``image2D t @rasterizer_ordered``, so subscripts become ``imageLoad`` and
``imageStore``, and each generator binds the view in its own raster-ordered
form (an HLSL or Slang ``RasterizerOrdered*`` view, a Metal raster order group,
a GLSL fragment-shader-interlock section).

:func:`import_rasterizer_ordered_views` rewrites each rasterizer-ordered type in
the Slang AST to its read-write view before the importer reads any type, and
marks the global declarations so the importer adds ``@rasterizer_ordered``.
"""

from __future__ import annotations

import re

from ..common_ast import AssignmentNode, VariableNode

__all__ = ["import_rasterizer_ordered_views", "is_rasterizer_ordered_declaration"]

# Each Slang rasterizer-ordered view and the read-write view it accesses like.
RASTERIZER_ORDERED_VIEWS = {
    "RasterizerOrderedTexture1D": "RWTexture1D",
    "RasterizerOrderedTexture1DArray": "RWTexture1DArray",
    "RasterizerOrderedTexture2D": "RWTexture2D",
    "RasterizerOrderedTexture2DArray": "RWTexture2DArray",
    "RasterizerOrderedTexture3D": "RWTexture3D",
    "RasterizerOrderedBuffer": "RWBuffer",
    "RasterizerOrderedStructuredBuffer": "RWStructuredBuffer",
    "RasterizerOrderedByteAddressBuffer": "RWByteAddressBuffer",
}

_VIEW_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(RASTERIZER_ORDERED_VIEWS, key=len, reverse=True)) + r")\b"
)

_MARK = "slang_rasterizer_ordered"


def _read_write_type(type_name):
    """``type_name`` with each rasterizer-ordered view replaced, or ``None`` if it has none."""
    if not isinstance(type_name, str) or "RasterizerOrdered" not in type_name:
        return None
    rewritten = _VIEW_PATTERN.sub(
        lambda match: RASTERIZER_ORDERED_VIEWS[match.group(1)], type_name
    )
    return rewritten if rewritten != type_name else None


def _rewrite_declaration(declaration, mark):
    if not isinstance(declaration, VariableNode):
        return
    rewritten = _read_write_type(getattr(declaration, "vtype", None))
    if rewritten is None:
        return
    declaration.vtype = rewritten
    if mark:
        setattr(declaration, _MARK, True)


def import_rasterizer_ordered_views(ast, functions=()):
    """Rewrite rasterizer-ordered types in ``ast`` to their read-write views.

    Global declarations, including exported ones, keep a mark that
    :func:`is_rasterizer_ordered_declaration` reads; function parameters take
    the read-write type only, because the resource they receive carries the
    qualifier.
    """
    globals_ = list(getattr(ast, "global_vars", []) or [])
    globals_ += [
        getattr(export, "item", None) for export in getattr(ast, "exports", []) or []
    ]
    for node in globals_:
        declaration = node.left if isinstance(node, AssignmentNode) else node
        _rewrite_declaration(declaration, mark=True)
    for function in functions:
        for param in getattr(function, "params", []) or []:
            _rewrite_declaration(param, mark=False)


def is_rasterizer_ordered_declaration(node):
    """Whether ``node`` was declared with a Slang rasterizer-ordered view."""
    return bool(getattr(node, _MARK, False))
