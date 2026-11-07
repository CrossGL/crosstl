"""Offsets of HLSL resource-pointer aliases, chosen by the source expression.

A CrossGL pointer into a buffer (``device const float2* row = root + k;``) lowers to the root resource plus an
offset local (``row_offset``), and an access through it renders as ``root[(row_offset + i)]``. The offset used to
be looked up by the rendered resource name. Every alias renders as its root, so a direct access to the root,
``root[i]``, also found the alias and became ``root[(row_offset + i)]``: a silent wrong-index read or write. With
two aliases of one root, an access through either took the first alias's offset.

The access's source expression settles it. A source identifier that names an alias has that alias's offset. A
source identifier that names anything else, including the root resource itself, has none.
"""

from __future__ import annotations

from ..ast import IdentifierNode

__all__ = ["hlsl_alias_offset_for_source", "hlsl_source_identifier_name"]


def hlsl_source_identifier_name(expr) -> str | None:
    """The name of ``expr`` when it is a plain identifier, else None."""
    if isinstance(expr, IdentifierNode) and isinstance(expr.name, str):
        return expr.name
    return None


def hlsl_alias_offset_for_source(codegen, source_name: str) -> str | None:
    """The offset local of the alias named ``source_name``, or None for a non-alias."""
    binding = codegen.current_hlsl_resource_pointer_aliases.get(source_name)
    if binding is None:
        return None
    if binding.get("kind") == "workgroup-pointer" and not binding.get(
        "is_pointer_alias"
    ):
        return None
    return binding.get("offset")
