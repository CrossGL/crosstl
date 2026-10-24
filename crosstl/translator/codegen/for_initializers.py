"""Render ``for`` initializers that declare or evaluate several items.

``for (short i = 0, j = 0; ...)`` reaches the code generators as a list of
``VariableNode`` declarations, and ``for (i = 0, j = 0; ...)`` as a list of
expressions. C-family targets accept one declaration whose declarators share
a type, or a comma expression. These helpers join per-item renderings into
that form and raise :class:`ForInitializerError` when the declarations do not
share one type spelling, so a generator never emits a partial initializer.
"""

import re
from typing import Callable, Sequence

__all__ = ["ForInitializerError", "join_for_initializer"]

# A declarator after the shared type: a name, optional array extents, then an
# initializer or nothing.
_DECLARATOR_RE = re.compile(r"[A-Za-z_]\w*\s*(?:\[[^\]]*\]\s*)*(?:=|$)")


class ForInitializerError(ValueError):
    """A ``for`` initializer list cannot be written as one declaration."""

    def __init__(self, reason: str):
        super().__init__(
            f"Cannot emit a for-loop initializer with several items: {reason}"
        )
        self.reason = reason


def _shared_type_prefix(declarations: Sequence[str]) -> str:
    prefix = declarations[0]
    for declaration in declarations[1:]:
        length = 0
        limit = min(len(prefix), len(declaration))
        while length < limit and prefix[length] == declaration[length]:
            length += 1
        prefix = prefix[:length]
    # The type ends at the last space before the first declarator name.
    cut = max(prefix.rfind(" "), prefix.rfind("*"), prefix.rfind("&"))
    return prefix[: cut + 1] if cut >= 0 else ""


def join_for_initializer(
    items: Sequence[object],
    is_declaration: Callable[[object], bool],
    render_declaration: Callable[[object], str],
    render_expression: Callable[[object], str],
) -> str:
    """Join a list initializer into one declaration or comma expression."""
    if not items:
        return ""
    declarations = [is_declaration(item) for item in items]
    if any(declarations) and not all(declarations):
        raise ForInitializerError("declarations mixed with expressions")
    if not all(declarations):
        return ", ".join(render_expression(item).strip().rstrip(";") for item in items)
    rendered = [render_declaration(item).strip().rstrip(";").strip() for item in items]
    if any(not text or "\n" in text or ";" in text for text in rendered):
        raise ForInitializerError("a declaration needs more than one statement")
    if len(rendered) == 1:
        return rendered[0]
    prefix = _shared_type_prefix(rendered)
    if not prefix.strip() or not all(
        _DECLARATOR_RE.match(text[len(prefix) :].lstrip()) for text in rendered
    ):
        raise ForInitializerError("the declarations do not share one type")
    return ", ".join(
        [rendered[0], *(text[len(prefix) :].lstrip() for text in rendered[1:])]
    )
