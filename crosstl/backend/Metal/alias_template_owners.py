"""Name member alias templates through their concrete owner struct.

A member alias template is often reached through a function-local alias of
its owner, as in

.. code-block:: cpp

    using CFrag = typename Tile_Frag::Frag_t;   // names struct Frag
    using cfrag_t = typename CFrag::template frag_t<float>;

Once the owner is concrete, ``CFrag::frag_t<float>`` denotes the type the
alias template names ([temp.alias]/2), here ``metal::vec<float, 4>``. The
importer expands member alias templates only where the owner is spelled by
its struct name, so the local spelling reached CrossGL as an undeclared type
and the argument types of calls such as ``CFrag::load(value, ...)`` could not
be inferred.

:func:`expose_alias_template_owners` replaces such a local owner spelling with
the struct's name, so the existing expansion applies. An owner is replaced
only outside template declarations, when the aliases visible at the name
resolve it to exactly one concrete struct, and when that struct declares the
member alias template; anything else keeps its spelling and its diagnostics.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

_TYPENAME = re.compile(r"^typename\s+")


def expose_alias_template_owners(
    preprocessor: Any, code: str, structs: Sequence[Any]
) -> str:
    """Return ``code`` with local owner aliases of member alias templates resolved."""
    owners_by_name: dict[str, list[Any]] = {}
    for struct in structs:
        owners_by_name.setdefault(struct.name, []).append(struct)
        if struct.qualified_name and struct.qualified_name != struct.name:
            owners_by_name.setdefault(struct.qualified_name, []).append(struct)
    members = sorted(
        {
            alias
            for struct in structs
            for alias in getattr(struct, "type_alias_templates", {}) or {}
        },
        key=lambda alias: (-len(alias), alias),
    )
    if not members or "::" not in code:
        return code
    pattern = re.compile(
        r"(?<![A-Za-z0-9_:])(?P<owner>[A-Za-z_]\w*)\s*::\s*(?:template\s+)?"
        rf"(?P<member>{'|'.join(re.escape(member) for member in members)})\s*<"
    )
    matches = [
        match
        for match in pattern.finditer(code)
        if match.group("owner") not in owners_by_name
    ]
    if not matches:
        return code
    template_spans = preprocessor._find_template_declaration_spans(code)
    ignored = sorted(
        [*template_spans, *preprocessor._find_comment_and_literal_spans(code)]
    )
    local_aliases = preprocessor._collect_local_type_alias_bindings(
        code, [(0, len(code))], skip_spans=template_spans
    )
    if not local_aliases:
        return code
    replacements: list[tuple[int, int, str]] = []
    for match in matches:
        if preprocessor._containing_span(match.start(), ignored) is not None:
            continue
        owner = _concrete_owner(
            preprocessor, match.group("owner"), match.start(), local_aliases, structs
        )
        candidates = owners_by_name.get(owner or "", [])
        if len(candidates) != 1 or match.group("member") not in (
            candidates[0].type_alias_templates or {}
        ):
            continue
        replacements.append((match.start("owner"), match.end("owner"), owner))
    if not replacements:
        return code
    return preprocessor._apply_text_replacements(code, replacements)


def _concrete_owner(
    preprocessor: Any,
    owner: str,
    position: int,
    local_aliases: dict,
    structs: Sequence[Any],
) -> str | None:
    if owner not in local_aliases:
        return None
    resolved = _TYPENAME.sub(
        "", preprocessor._resolve_type_aliases_at(owner, local_aliases, position)
    ).lstrip(":")
    if "::" in resolved:
        # The local alias may name a member alias of a concrete struct, as in
        # ``using CFrag = typename Tile_Frag::Frag_t;``.
        resolved = _TYPENAME.sub(
            "",
            preprocessor._normalize_template_argument_text(
                preprocessor._canonicalize_qualified_struct_type_aliases(
                    f"typename {resolved}", list(structs)
                )
            ),
        ).lstrip(":")
    return resolved or None
