"""Resolve qualified struct member type names inside concrete Metal functions.

A concrete function may name a struct through a nested type alias, for example
``thread typename kernel_t::loader_t loader(src, 2);`` where ``kernel_t`` is a
function-local ``using`` alias of a materialized struct and ``loader_t`` is a
type alias declared inside that struct.  The struct-method lowering only reads
plain type names, so such a receiver would not resolve.  This pass rewrites a
qualified name to the concrete struct it denotes when, and only when, every
component resolves through visible aliases and struct-scoped type aliases.
Anything else is left untouched for the existing diagnostics.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

_QUALIFIED_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9_:])(?:typename\s+)?(?:::)?[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)+"
    r"(?![\w:<(])"
)
_ALIAS_DECLARATION_RE = re.compile(
    r"\b(?:using\s+[A-Za-z_]\w*\s*=|typedef\b)[^;{}]*;", re.DOTALL
)


def resolve_qualified_struct_member_types(
    preprocessor: Any,
    code: str,
    structs: Sequence[Any],
    type_aliases: Any,
    excluded_spans: Sequence[tuple[int, int]],
) -> str:
    """Return ``code`` with resolvable qualified struct type names replaced.

    Only text inside non-template function bodies is considered, comments,
    literals and local ``using``/``typedef`` alias declarations (which the
    alias bindings already resolve) are skipped, and a replacement is made only
    when the qualified name resolves to the name of one of ``structs``.  A qualified name followed
    by ``(`` or ``<`` is a call or template-id and is never rewritten.
    """
    if "::" not in code or not structs:
        return code
    struct_names = {struct.name for struct in structs}
    if not any(struct.type_aliases for struct in structs):
        return code
    ignored = preprocessor._find_comment_and_literal_spans(code)
    replacements: list[tuple[int, int, str]] = []
    for function in preprocessor._find_non_template_function_definitions(
        code, list(excluded_spans)
    ):
        body_start, body_end = function.body_span
        alias_spans = [
            alias.span()
            for alias in _ALIAS_DECLARATION_RE.finditer(code, body_start, body_end)
        ]
        for match in _QUALIFIED_NAME_RE.finditer(code, body_start, body_end):
            if preprocessor._containing_span(match.start(), ignored) is not None:
                continue
            if preprocessor._containing_span(match.start(), alias_spans) is not None:
                continue
            text = match.group(0)
            qualified = re.sub(r"^typename\s+", "", text)
            resolved = preprocessor._resolve_type_aliases_at(
                qualified, type_aliases, match.start()
            )
            resolved = preprocessor._canonicalize_qualified_struct_type_aliases(
                resolved, list(structs)
            )
            resolved = preprocessor._normalize_template_argument_text(resolved)
            if resolved in struct_names and resolved != qualified:
                replacements.append((match.start(), match.end(), resolved))
    if not replacements:
        return code
    return preprocessor._apply_text_replacements(code, replacements)
