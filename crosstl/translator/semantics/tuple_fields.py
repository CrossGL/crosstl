"""Refuse tuple field access, ``t.1``.

CrossGL reads ``t.1`` as field 1 of the tuple ``t``, the way Rust and Mojo write it (the Rust Reference, "Tuple
indexing expressions"). No CrossGL generator lowers a tuple to a target type yet: the tuple literal ``(a, b,)``
reaches them as an untyped list, which C-family targets would print as a braced list and a numbered member that
does not exist. :func:`refuse_tuple_fields` therefore refuses every tuple field access with
:class:`TupleFieldUnsupportedError` before code generation, so no target writes ``t.1`` as text.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .rewrite import child_slots, get_child

__all__ = ["TupleFieldUnsupportedError", "is_tuple_field", "refuse_tuple_fields"]


class TupleFieldUnsupportedError(TranslationDiagnostic):
    """A tuple field access, which no generator lowers."""

    def __init__(self, target, field, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = ("tuple values",)
        super().__init__(
            f"tuple field access '.{field}' is not supported: no {target or 'CrossGL'} lowering exists for "
            "tuple values",
            target=target,
            code=f"{CODE_PREFIX}{target or 'crossgl'}-tuple-field",
        )


def is_tuple_field(node) -> bool:
    """Whether ``node`` reads a numbered tuple field such as ``t.0``."""
    if not isinstance(node, A.MemberAccessNode):
        return False
    member = node.member
    return isinstance(member, str) and member.isascii() and member.isdigit()


def refuse_tuple_fields(program, target) -> None:
    """Raise :class:`TupleFieldUnsupportedError` at the first tuple field access in ``program``."""
    target = str(target or "")
    pending = [program]
    while pending:
        node = pending.pop()
        if not isinstance(node, A.ASTNode):
            continue
        if is_tuple_field(node):
            raise TupleFieldUnsupportedError(
                target,
                node.member,
                source_location=getattr(node, "source_location", None),
            )
        children = [get_child(node, field, key) for field, key in child_slots(node)]
        pending.extend(reversed(children))
