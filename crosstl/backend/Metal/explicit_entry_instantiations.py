"""Explicit instantiations of entry-point templates that reach the parser unmaterialized.

``template [[host_name("k")]] [[kernel]] decltype(f<float>) f<float>;`` asks for a concrete kernel. The preprocessor
replaces each such declaration with the materialized kernel, but it leaves the source unchanged when the source asks
for more instantiations than ``max_template_specializations`` allows. The parser then skipped every declaration, and
the translation had no entry point at all.

The parser records each explicit instantiation of an entry point it skips (:func:`instantiated_entry_name`). Once the
unit is parsed, :func:`refuse_unmaterialized_entry_instantiations` refuses the unit when one of them names a function
that is still a template. An instantiation of a function the source already defines concretely is redundant and is
still skipped.
"""

from __future__ import annotations

import re

from .preprocessor import MetalTemplateSpecializationError

__all__ = ["instantiated_entry_name", "refuse_unmaterialized_entry_instantiations"]

_ENTRY_ATTRIBUTE = re.compile(
    r"^\[\[\s*(?:host_name\s*\(|(?:kernel|vertex|fragment|mesh|object)\s*(?:\]\]|,))"
)
_ENTRY_KEYWORDS = frozenset({"kernel", "vertex", "fragment"})


def _declaration_tokens(tokens, start):
    """The tokens from ``start`` to the declaration's semicolon at depth zero, with their depths."""
    depth = 0
    for token in tokens[start:]:
        kind = token[0]
        if kind == "EOF" or (kind == "SEMICOLON" and depth == 0):
            return
        if kind in {"RPAREN", "RBRACKET", "RBRACE"}:
            depth = max(0, depth - 1)
        yield token, depth
        if kind in {"LPAREN", "LBRACKET", "LBRACE"}:
            depth += 1


def instantiated_entry_name(tokens, start):
    """``(name, spelling)`` of the explicit entry-point instantiation that starts after ``template`` at ``start``.

    ``None`` when the declaration does not declare an entry point or names no template-id.
    """
    declaration = list(_declaration_tokens(tokens, start))
    if not any(
        (token[0] == "ATTRIBUTE" and _ENTRY_ATTRIBUTE.match(str(token[1])))
        or str(token[1]) in _ENTRY_KEYWORDS
        for token, _depth in declaration
    ):
        return None
    name = None
    for (token, depth), (following, _following_depth) in zip(
        declaration, declaration[1:]
    ):
        if depth == 0 and token[0] == "IDENTIFIER" and following[0] == "LESS_THAN":
            name = str(token[1])
    if name is None:
        return None
    spelled = " ".join(str(token[1]) for token, _depth in declaration)
    return name, spelled[:200]


def refuse_unmaterialized_entry_instantiations(instantiations, functions, limit):
    """Raise when a recorded instantiation names a function of ``functions`` that is still a template."""
    templates = {
        getattr(function, "name", None)
        for function in functions or ()
        if getattr(function, "template_parameters", None)
        or getattr(function, "generics", None)
    }
    for name, spelled in instantiations:
        if name not in templates:
            continue
        suggested_action = "raise max_template_specializations for this source or translate it as a project"
        raise MetalTemplateSpecializationError(
            "Metal explicit entry-point instantiation was not materialized: "
            f"'template {spelled}'. The source declares more explicit instantiations than the "
            f"specialization limit {limit} allows, and an unmaterialized kernel would be dropped. "
            f"Suggested action: {suggested_action}.",
            limit=limit,
            requested_signature=spelled,
            suggested_action=suggested_action,
        )
