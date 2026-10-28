"""Lower conditionals on structures, arrays and matrices to ``if``/``else``, for WGSL.

WGSL output writes ``c ? a : b`` as ``select(b, a, c)``, and ``select`` takes scalars and vectors only (WGSL
section 17.3), so a conditional whose branches are structures, arrays or matrices has no expression form. When such
a conditional is the whole initializer of a local declaration or the whole value of an ``=`` assignment, the
statement becomes a declaration without an initializer followed by an ``if``/``else`` that assigns the chosen
branch::

    Pair r = c ? p : q;        var r: Pair;
                               if (c) { r = p; } else { r = q; }

The condition is evaluated once and only the chosen branch is evaluated, as in the C-family source. A conditional in
a branch is lowered again as that branch's assignment. Anywhere else, such as ``(c ? p : q).a``, the conditional is
refused with ``ValueError`` rather than written as a ``select`` that naga rejects.
"""

from __future__ import annotations

import copy

from ..ast import (
    AssignmentNode,
    BlockNode,
    ExpressionStatementNode,
    IdentifierNode,
    IfNode,
    SwizzleNode,
    TernaryOpNode,
    VariableNode,
)

SELECT_SCALAR_TYPES = frozenset({"bool", "i32", "u32", "f32", "f16"})


class WGSLCompositeConditionalMixin:
    """Statement lowering for conditionals that ``select`` cannot take; see the module docstring."""

    def is_select_operand_type(self, vtype):
        """Whether WGSL ``select`` takes values of ``vtype``.

        An unknown type counts as one, so a conditional whose type cannot be inferred keeps its ``select``.
        """
        if vtype is None:
            return True
        try:
            type_name = self.type_name_string(vtype)
        except ValueError:
            return True
        if type_name is None or type_name in SELECT_SCALAR_TYPES:
            return True
        return self.vector_component_count(vtype) is not None

    def lower_composite_ternary(self, stmt):
        """The statements that replace ``T x = c ? a : b`` or ``x = c ? a : b`` on a composite ``T``, or None."""
        if (
            isinstance(stmt, ExpressionStatementNode)
            and isinstance(stmt.expression, AssignmentNode)
            and not isinstance(stmt.expression.target, SwizzleNode)
        ):
            stmt = stmt.expression
        if isinstance(stmt, VariableNode):
            value = stmt.initial_value
            if not isinstance(value, TernaryOpNode):
                return None
            var_type = self.local_variable_declaration_type(stmt)
            if self.is_select_operand_type(var_type):
                return None
            declaration = copy.copy(stmt)
            declaration.var_type = var_type
            declaration.vtype = var_type
            declaration.initial_value = None
            declaration.is_mutable = True
            return [declaration, AssignmentNode(IdentifierNode(stmt.name), value)]
        if isinstance(stmt, AssignmentNode):
            value = stmt.value
            if stmt.operator != "=" or not isinstance(value, TernaryOpNode):
                return None
            if self.is_select_operand_type(self.expression_type(stmt.target)):
                return None
            return [
                IfNode(
                    value.condition,
                    BlockNode([AssignmentNode(stmt.target, value.true_expr)]),
                    BlockNode([AssignmentNode(stmt.target, value.false_expr)]),
                )
            ]
        return None

    def require_select_operands(self, expr):
        """Refuse the conditional ``expr`` when ``select`` cannot take its branches."""
        for branch in (expr.true_expr, expr.false_expr):
            branch_type = self.expression_type(branch)
            if self.is_select_operand_type(branch_type):
                continue
            raise ValueError(
                "WGSL target cannot lower a conditional expression of type "
                f"{self.type_name_string(branch_type)} here: WGSL select takes "
                "scalars and vectors only, and a struct, array or matrix "
                "conditional is lowered only as a whole initializer or "
                "assignment value"
            )
