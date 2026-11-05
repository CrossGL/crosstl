"""Visibility of namespace-scoped Metal type aliases in reopened namespace definitions.

A namespace may be defined in several blocks, and a name declared in one block is visible in every later block of
the same namespace and of the namespaces nested in it (C++ [namespace.def], [basic.lookup.unqual]). The Metal
importer appends concrete template specializations inside reopened definitions of the template's namespace, so a
block-scoped alias inliner must also replace an alias's uses in those later blocks.

Block spans are ``(open, close + 1)`` brace offsets, and namespace spans are ``(open + 1, close, name)`` as the
Metal preprocessor reports them.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "alias_visible_at",
    "alias_visible_in_scope",
    "reopened_namespace_scopes",
]


def reopened_namespace_scopes(
    scope_span: tuple[int, int],
    declaration_end: int,
    namespace_spans: Sequence[tuple[int, int, str]],
) -> tuple[tuple[int, int], ...]:
    """Later blocks of the namespace whose body is ``scope_span``, as block spans in source order."""

    names = [
        name
        for start, end, name in namespace_spans
        if start == scope_span[0] + 1 and end == scope_span[1] - 1
    ]
    if not names or not names[0]:
        return ()
    namespace = names[0]
    nested = namespace + "::"
    return tuple(
        sorted(
            (start - 1, end + 1)
            for start, end, name in namespace_spans
            if start > declaration_end
            and (name == namespace or name.startswith(nested))
        )
    )


def alias_visible_at(alias: Mapping[str, Any], position: int) -> bool:
    """Whether an unqualified use at ``position`` names ``alias``."""

    if int(alias["end"]) <= position < int(alias["scope_end"]):
        return True
    return any(
        start < position < end for start, end in alias.get("reopened_scopes", ())
    )


def alias_visible_in_scope(
    alias: Mapping[str, Any], position: int, scope_end: int
) -> bool:
    """Whether ``alias`` is visible throughout the block that ends at ``scope_end`` and contains ``position``."""

    if int(alias["end"]) <= position and int(alias["scope_end"]) >= scope_end:
        return True
    return any(
        start < position and end >= scope_end
        for start, end in alias.get("reopened_scopes", ())
    )
