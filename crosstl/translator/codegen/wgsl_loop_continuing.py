"""WGSL lowering of loop headers and steps that WGSL's ``for`` cannot hold, and of ``do``-``while``.

A WGSL ``for`` header holds at most one initializer, a variable declaration or an updating statement, and at
most one update, an updating statement or a function call; WGSL has no comma operator and no ``do``-``while``
statement (WGSL "for Statement", "loop Statement"). C-family sources write ``for (int i = 0, n = 8; i < n;
i++, n--)``, ``for (k = 0, m = 3; ...)``, ``for (...; ...; data[i++] = v)`` and ``do { ... } while (c);``.

- Several initializers become statements of a block that encloses the loop, run in source order, so their
  names still go out of scope with the loop.
- Several updates, or one whose operands have side effects, become statements of the ``continuing`` block of
  a ``loop`` (WGSL "continuing Statement"). The loop starts with ``if (!(c)) { break; }`` for the condition.
  A ``continue`` in the body jumps to the ``continuing`` block, as it jumps to the update in C, GLSL, HLSL
  and MSL.
- ``do body while (c);`` becomes ``loop { body continuing { break if !(c); } }`` (WGSL "break if
  Statement"). ``continue`` again reaches the test, and side effects of ``c`` run in the ``continuing``
  block before it.

The body stays a nested compound statement, so the ``continuing`` block never sees a declaration of the body,
which WGSL forbids when a ``continue`` can skip it.
"""

from __future__ import annotations

from ..ast import (
    BlockNode,
    CommaExpressionNode,
    ExpressionStatementNode,
    StatementNode,
    UnaryOpNode,
    VariableNode,
)
from .wgsl_value_side_effects import statement_operand_side_effects


class WGSLContinuingLoopNode(StatementNode):
    """A WGSL ``loop`` with a ``continuing`` block; only the WGSL emitter creates and consumes it.

    ``body`` is the list of statements before the ``continuing`` block, ``continuing`` the statements in it
    and ``break_if`` the expression of its final ``break if``, or None.
    """

    def __init__(self, body, continuing, break_if=None, **kwargs):
        super().__init__(**kwargs)
        self.body = list(body)
        self.continuing = list(continuing)
        self.break_if = break_if

    def __repr__(self):
        return "WGSLContinuingLoopNode()"


def for_header_items(value):
    """The expressions of a for-loop initializer or update, with comma operators and lists flattened."""
    pending = list(value) if isinstance(value, (list, tuple)) else [value]
    items = []
    while pending:
        item = pending.pop(0)
        if item is None:
            continue
        if isinstance(item, ExpressionStatementNode) and isinstance(
            item.expression, CommaExpressionNode
        ):
            item = item.expression
        if isinstance(item, CommaExpressionNode):
            pending[:0] = list(item.expressions)
            continue
        items.append(item)
    return items


def as_statement(item):
    """``item`` as a statement: declarations and statements as they are, an expression wrapped."""
    if isinstance(item, (VariableNode, StatementNode)):
        return item
    return ExpressionStatementNode(item)


class WGSLLoopContinuingMixin:
    """Loop lowering and emission for WGSL; see the module docstring.

    Uses ``statement_operand_side_effects``, ``loop_guard`` and ``hoist_value_side_effects`` from
    :class:`~crosstl.translator.codegen.wgsl_value_side_effects.WGSLValueSideEffectsMixin`.
    """

    def for_initializer_statements(self, init):
        """The statements that run a for-loop initializer of several items, or None when it has one."""
        inner = init.expression if isinstance(init, ExpressionStatementNode) else init
        if not isinstance(inner, (list, tuple, CommaExpressionNode)):
            return None
        return [as_statement(item) for item in for_header_items(init)]

    def for_update_needs_continuing(self, update):
        """True when a for-loop update is several items, or one whose operands have side effects."""
        items = for_header_items(update)
        return len(items) > 1 or any(
            statement_operand_side_effects(item) for item in items
        )

    def lower_for_to_continuing_loop(self, stmt):
        """``for (init; c; u1, u2) body`` as ``{ init; loop { if (!(c)) { break; } body continuing { u1; u2; } } }``."""
        prelude = self.for_initializer_statements(stmt.init)
        if prelude is None:
            prelude = [] if stmt.init is None else [as_statement(stmt.init)]
        body = []
        if stmt.condition is not None:
            body.extend(self.loop_guard(stmt.condition))
        if stmt.body is not None:
            body.append(stmt.body)
        continuing = [as_statement(item) for item in for_header_items(stmt.update)]
        loop = WGSLContinuingLoopNode(body, continuing)
        if not prelude:
            return [loop]
        return [BlockNode([*prelude, loop])]

    def lower_do_while_statement(self, stmt):
        """``do body while (c);`` as ``loop { body continuing { break if !(c); } }``."""
        continuing = []
        condition = self.hoist_value_side_effects(stmt.condition, continuing)
        body = [] if stmt.body is None else [stmt.body]
        return [
            WGSLContinuingLoopNode(
                body, continuing, break_if=UnaryOpNode("!", condition)
            )
        ]

    def generate_continuing_loop(self, node, indent):
        pad = "    " * indent
        inner = pad + "    "
        lines = [f"{pad}loop {{"]
        self.push_identifier_scope()
        try:
            for statement in node.body:
                if isinstance(statement, BlockNode) and not statement.statements:
                    continue
                lines.append(self.generate_statement(statement, indent + 1))
            if node.continuing or node.break_if is not None:
                lines.append(f"{inner}continuing {{")
                self.push_identifier_scope()
                try:
                    for statement in node.continuing:
                        lines.append(self.generate_statement(statement, indent + 2))
                    if node.break_if is not None:
                        condition = self.generate_expression(node.break_if)
                        lines.append(f"{inner}    break if {condition};")
                finally:
                    self.pop_identifier_scope()
                lines.append(f"{inner}}}")
        finally:
            self.pop_identifier_scope()
        lines.append(f"{pad}}}")
        return "\n".join(lines)
