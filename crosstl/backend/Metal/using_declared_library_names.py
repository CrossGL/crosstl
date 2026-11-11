"""Metal library functions a using-declaration adds to unqualified lookup.

``using metal::max;`` at namespace scope makes ``metal::max`` a member of that scope's overload set for unqualified
calls ([namespace.udecl]/1). A source that also declares ``bfloat max(bfloat, bfloat)`` there has one overload set
for ``max``: ``max(h0, h1)`` on ``half`` values selects the library's exact match, and only ``bfloat`` arguments
select the source function ([over.match.best]). The importer resolves calls through the library-collision path for
such names, as it already does for a name the source calls qualified (``metal::max``).
"""

from __future__ import annotations

__all__ = ["using_declared_alias_targets", "using_declared_library_names"]


def using_declared_library_names(ast, library_names) -> set[str]:
    """Names of library functions in ``library_names`` that a global using-declaration brings into scope."""
    names = set()
    for declaration in getattr(ast, "using_declarations", None) or []:
        if declaration.get("namespace"):
            continue
        target = str(declaration.get("target") or "").lstrip(":")
        scope, _, name = target.rpartition("::")
        if scope == "metal" and name in library_names:
            names.add(name)
    return names


def using_declared_alias_targets(declarations, scope, name, normalize) -> list[str]:
    """Qualified names that using-declarations in namespace ``scope`` introduce as ``name``.

    ``using c10::metal::opmath_t;`` in the global namespace makes ``opmath_t`` name ``c10::metal::opmath_t`` there
    ([namespace.udecl]/1), so unqualified type lookup in that scope finds the alias template or struct it names.
    """
    targets = []
    for declaration in declarations or []:
        if normalize(declaration.get("namespace") or "") != scope:
            continue
        target = normalize(str(declaration.get("target") or "")).lstrip(":")
        if target.rpartition("::")[2] == name and "::" in target:
            if target not in targets:
                targets.append(target)
    return targets
