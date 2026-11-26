"""The scope of the declarations a Slang ``switch`` case makes.

A ``case`` label opens no scope in Slang, HLSL or C: every case of a ``switch`` shares the braces of the ``switch``
body. Sources that declare a local of the same name in two cases (``case 0: { float t; ... } case 1: { int t; ... }``)
give each case its own braces, and CrossGL keeps the case's statements without them, so the Slang generator writes
the braces back. A case is braced when it declares a local at its top level and no later case reads that local, as
a case that falls through into the next one may share it.
"""

from __future__ import annotations

from ..ast import BlockNode, CaseNode, VariableNode
from .specialization_constants import identifier_names

__all__ = ["slang_case_needs_scope"]


def _top_level_declarations(statements):
    return {
        statement.name
        for statement in statements
        if isinstance(statement, VariableNode) and isinstance(statement.name, str)
    }


def _free_names(generator, statements):
    """The names ``statements`` read that none of their own earlier declarations or nested blocks declares.

    A name declared inside a nested ``if`` or loop body counts as free, which keeps a case unbraced: the old,
    shared scope.
    """
    declared, free = set(), set()
    for statement in statements:
        if isinstance(statement, BlockNode):
            free |= (
                _free_names(generator, generator.get_statements(statement)) - declared
            )
        elif isinstance(statement, VariableNode):
            initial = getattr(
                statement, "initial_value", getattr(statement, "value", None)
            )
            free |= identifier_names(initial) - declared
            if isinstance(statement.name, str):
                declared.add(statement.name)
        else:
            free |= identifier_names(statement) - declared
    return free


def slang_case_needs_scope(generator, cases, position):
    """Whether case ``position`` of ``cases`` is written inside braces of its own."""
    declared = _top_level_declarations(
        generator.get_statements(cases[position].statements)
    )
    if not declared:
        return False
    for later in cases[position + 1 :]:
        if not isinstance(later, CaseNode):
            continue
        statements = generator.get_statements(later.statements)
        # A later case that reads one of these locals before declaring its own shares it through fall-through.
        if declared & _free_names(generator, statements):
            return False
        declared -= _top_level_declarations(statements)
        if not declared:
            break
    return True
