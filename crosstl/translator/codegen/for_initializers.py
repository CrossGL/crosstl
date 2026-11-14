"""For-loop initializers that declare several variables: ``for (int a = 0, b = n; a < b; a++, b--)``.

The CrossGL parser keeps such an initializer as a list of declarations, or of expressions for
``for (a = 0, b = n; ...)``. GLSL 4.60 section 6.3 takes one declaration there, whose init-declarator-list names
several variables of one type; C++17 [stmt.for] and [dcl.dcl], which MSL and Slang follow, take one
simple-declaration the same way. An emitter generates each item as it would alone and :func:`join_for_initializer`
writes them as that one declaration, or as one comma expression.
"""

from __future__ import annotations

import re

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["ForInitializerError", "join_for_initializer"]

_DECLARATOR = re.compile(
    r"^(?P<prefix>.*?)(?P<name>[A-Za-z_]\w*)(?P<suffix>(\s*\[[^\]]*\])*)\s*$"
)


class ForInitializerError(TranslationDiagnostic):
    """A for-loop initializer list that one declaration or one comma expression cannot write."""

    code = f"{CODE_PREFIX}for-initializer-unsupported"


def _split_declaration(text):
    """``("int ", "a = 0")`` for ``int a = 0``: the declaration's type and qualifiers, and its declarator."""
    head, equals, initializer = text.partition("=")
    match = _DECLARATOR.match(head.rstrip())
    if match is None or not match.group("prefix").strip():
        return None
    declarator = match.group("name") + match.group("suffix")
    if equals:
        declarator = f"{declarator} ={initializer}"
    return match.group("prefix"), declarator


def join_for_initializer(parts, declarations, target):
    """One initializer for the generated ``parts``; ``declarations[i]`` says whether part ``i`` declares a variable.

    Raises :class:`ForInitializerError` when the items mix declarations with expressions, declare variables of
    different types, or one of them needs more than one statement.
    """
    parts = [str(part).strip().rstrip(";").strip() for part in parts]
    if any(";" in part for part in parts):
        raise ForInitializerError(
            "a for-loop initializer item needs more than one statement", target=target
        )
    if not any(declarations):
        return ", ".join(parts)
    if not all(declarations):
        raise ForInitializerError(
            "a for-loop initializer mixes declarations and expressions", target=target
        )
    split = [_split_declaration(part) for part in parts]
    if any(item is None for item in split):
        raise ForInitializerError(
            "a for-loop initializer declaration has no type to share", target=target
        )
    prefixes = {prefix.strip() for prefix, _ in split}
    if len(prefixes) != 1:
        raise ForInitializerError(
            "a for-loop initializer declares variables of different types "
            f"({', '.join(sorted(prefixes))}); one declaration has one type",
            target=target,
        )
    return split[0][0] + ", ".join(declarator for _, declarator in split)
