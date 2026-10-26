"""WGSL switch clauses built from C-style switch cases.

HLSL, GLSL, MSL and CrossGL ``switch`` statements follow C: the case labels are entry points into one statement
list. Consecutive labels share the body that follows them, and a body that does not end in a jump falls through into
the next one. A WGSL switch clause instead runs exactly one compound statement and never falls through, but its case
selectors may list several values as well as ``default`` (WGSL, "Switch Statement"). Every WGSL switch also needs
exactly one ``default`` selector.

:func:`wgsl_switch_clauses` therefore gives each WGSL clause every label that leads into the same body, and appends
to a body that can fall through the statements it falls into. Emission order follows the source, so the output does
not depend on hashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ast import BlockNode, BreakNode, ContinueNode, ReturnNode

__all__ = [
    "WGSLSwitchClause",
    "ends_in_jump",
    "is_discard_statement",
    "wgsl_switch_clauses",
]

DEFAULT_SELECTOR = None


@dataclass
class WGSLSwitchClause:
    """One WGSL switch clause: its selectors (``None`` is ``default``) and its statements."""

    selectors: list = field(default_factory=list)
    statements: list = field(default_factory=list)

    @property
    def is_default(self):
        return any(selector is DEFAULT_SELECTOR for selector in self.selectors)


def _case_statements(case):
    if case is None:
        return []
    if isinstance(case, BlockNode):
        return list(case.statements or [])
    if hasattr(case, "statements"):
        return list(getattr(case, "statements") or [])
    if hasattr(case, "body"):
        body = getattr(case, "body")
        if isinstance(body, BlockNode):
            return list(body.statements or [])
        return list(body or [])
    if isinstance(case, list):
        return list(case)
    return [case]


def _source_labels(node):
    """``(value, statements)`` for every label in source order; ``value`` is ``None`` for ``default``."""
    labels = []
    has_default = False
    for case in getattr(node, "cases", None) or []:
        value = getattr(case, "value", None)
        has_default = has_default or value is None
        labels.append((value, _case_statements(case)))
    default_case = getattr(node, "default_case", None)
    if default_case is not None and not has_default:
        labels.append((DEFAULT_SELECTOR, _case_statements(default_case)))
    return labels


def is_discard_statement(statement):
    """Whether ``statement`` is a fragment ``discard`` (an identifier or a call without arguments)."""
    expression = getattr(statement, "expression", statement)
    callee = getattr(expression, "function", None) or getattr(expression, "name", None)
    name = callee if isinstance(callee, str) else getattr(callee, "name", None)
    arguments = getattr(expression, "arguments", None) or getattr(
        expression, "args", None
    )
    return name == "discard" and not arguments


def ends_in_jump(statements):
    """Whether control can never leave ``statements`` by reaching their end.

    Only a trailing ``break``, ``continue``, ``return`` or ``discard`` (directly or as the last statement of a nested
    block) counts. Anything else is treated as able to fall through, which keeps the lowering conservative: the
    statements appended after a jump the analysis cannot see are never executed.
    """
    if not statements:
        return False
    last = statements[-1]
    if isinstance(last, (BreakNode, ContinueNode, ReturnNode)):
        return True
    if isinstance(last, BlockNode):
        return ends_in_jump(list(last.statements or []))
    return is_discard_statement(last)


def wgsl_switch_clauses(node):
    """The WGSL clauses of switch ``node``, in source order, with exactly one ``default`` selector."""
    labels = _source_labels(node)
    groups = []
    pending = []
    for value, statements in labels:
        pending.append(value)
        if statements:
            groups.append(WGSLSwitchClause(selectors=pending, statements=statements))
            pending = []
    if pending:
        # Labels after the last body run nothing and leave the switch.
        groups.append(WGSLSwitchClause(selectors=pending, statements=[]))

    # A body that falls through runs every body after it up to the first one that ends in a jump.
    for index in range(len(groups) - 2, -1, -1):
        if not ends_in_jump(groups[index].statements):
            groups[index].statements = (
                groups[index].statements + groups[index + 1].statements
            )

    if not any(group.is_default for group in groups):
        groups.append(
            WGSLSwitchClause(selectors=[DEFAULT_SELECTOR], statements=[BreakNode()])
        )
    return groups
