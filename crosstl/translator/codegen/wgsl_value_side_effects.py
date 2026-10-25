"""Lower increments, decrements and assignments that are used as values, for WGSL.

WGSL has increment, decrement and assignment *statements* and no matching
expressions (WGSL sections 9.2 and 9.3). C-family sources use all three as
values: ``a[i++] = v``, ``int b = ++i``, ``x = y = 0``, ``while (n-- > 0)``.

``lower_value_side_effects`` rewrites one statement into a list of statements.
The side effects run first, in left-to-right order, and the rest of the
statement then reads the updated l-values, or a ``let`` copy of the prior value
for a postfix operator. Reads that the source leaves unsequenced against those
side effects (``i + i++``) are undefined in C, GLSL and HLSL, so any order is
faithful. Conditional evaluation is not: a side effect in the right operand of
``&&`` or ``||``, in a ternary branch, or in a ``for`` update raises
``ValueError`` rather than running unconditionally. So does a side effect whose
l-value calls a function, because the lowering evaluates that l-value twice.

A ``while`` condition is lowered to ``loop { ...; if (!c) { break; } body }``
and a ``for`` condition to an empty condition with the same guard at the top of
the body, so the side effects still run once per test.
"""

from __future__ import annotations

import copy

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CastNode,
    ConstructorNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LambdaNode,
    LiteralNode,
    LoopNode,
    MemberAccessNode,
    ReturnNode,
    SwitchNode,
    SwizzleNode,
    TernaryOpNode,
    TypeNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
    iter_ast_child_nodes,
)

INCREMENT_OPERATORS = ("++", "--")
SHORT_CIRCUIT_OPERATORS = ("&&", "||")
PRIOR_VALUE_PREFIX = "_crosstl_prior_"
# Node types an l-value may be built from when the lowering evaluates it twice.
REPEATABLE_NODE_TYPES = (
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    ArrayAccessNode,
    SwizzleNode,
    BinaryOpNode,
    UnaryOpNode,
    CastNode,
    TypeNode,
)


def is_increment(node):
    return isinstance(node, UnaryOpNode) and node.operator in INCREMENT_OPERATORS


def is_value_side_effect(node):
    return is_increment(node) or isinstance(node, AssignmentNode)


def contains_value_side_effect(value):
    """True when ``value`` or a descendant increments, decrements or assigns.

    Nodes keep legacy alias fields, so the walk visits each node once.
    """
    stack = list(iter_ast_child_nodes(value))
    visited = set()
    while stack:
        node = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if is_value_side_effect(node):
            return True
        if not isinstance(node, LambdaNode):
            stack.extend(node.child_nodes())
    return False


def statement_operand_side_effects(expr):
    """True when a statement-level expression has side effects in a value position."""
    if isinstance(expr, ExpressionStatementNode):
        expr = expr.expression
    if expr is None:
        return False
    if is_increment(expr):
        return contains_value_side_effect(expr.operand)
    if isinstance(expr, AssignmentNode):
        return contains_value_side_effect(expr.target) or contains_value_side_effect(
            expr.value
        )
    return contains_value_side_effect(expr)


def describe(node):
    if isinstance(node, IdentifierNode):
        return f"'{node.name}'"
    return type(node).__name__


def replaced(node, **fields):
    """A shallow copy of ``node`` with ``fields`` and their legacy aliases set."""
    aliases = {
        (AssignmentNode, "target"): "left",
        (AssignmentNode, "value"): "right",
        (FunctionCallNode, "arguments"): "args",
        (MemberAccessNode, "object_expr"): "object",
        (ArrayAccessNode, "array_expr"): "array",
        (ArrayAccessNode, "index_expr"): "index",
        (IfNode, "condition"): "if_condition",
        (IfNode, "else_branch"): "else_body",
    }
    result = copy.copy(node)
    for name, value in fields.items():
        setattr(result, name, value)
        alias = aliases.get((type(node), name))
        if alias is not None and hasattr(result, alias):
            setattr(result, alias, value)
    return result


class WGSLValueSideEffectsMixin:
    """Statement lowering for side effects used as values; see the module docstring."""

    def lower_value_side_effects(self, stmt):
        """The statements that replace ``stmt``, or None when it needs no lowering."""
        if isinstance(stmt, ExpressionStatementNode):
            if not statement_operand_side_effects(stmt.expression):
                return None
            prelude = []
            expression = self.lower_statement_expression(stmt.expression, prelude)
            if expression is None:
                return prelude
            return [*prelude, replaced(stmt, expression=expression)]
        if isinstance(stmt, AssignmentNode):
            if not statement_operand_side_effects(stmt):
                return None
            prelude = []
            assignment = self.lower_statement_expression(stmt, prelude)
            return [*prelude, assignment]
        if isinstance(stmt, VariableNode):
            return self.lower_value_field(stmt, "initial_value")
        if isinstance(stmt, ReturnNode):
            return self.lower_value_field(stmt, "value")
        if isinstance(stmt, SwitchNode):
            return self.lower_value_field(stmt, "expression")
        if isinstance(stmt, IfNode):
            return self.lower_if_statement(stmt)
        if isinstance(stmt, WhileNode):
            return self.lower_while_statement(stmt)
        if isinstance(stmt, ForNode):
            return self.lower_for_statement(stmt)
        return None

    def lower_value_field(self, stmt, field_name):
        value = getattr(stmt, field_name, None)
        if not contains_value_side_effect(value):
            return None
        prelude = []
        lowered = self.hoist_value_side_effects(value, prelude)
        return [*prelude, replaced(stmt, **{field_name: lowered})]

    def lower_statement_expression(self, expr, prelude):
        """``expr`` at statement level, with side effects of its operands hoisted.

        Returns None when nothing observable remains once they are hoisted.
        """
        if is_increment(expr):
            operand = self.hoist_value_side_effects(expr.operand, prelude)
            return replaced(expr, operand=operand)
        if isinstance(expr, AssignmentNode):
            target = self.hoist_value_side_effects(expr.target, prelude)
            value = self.hoist_value_side_effects(expr.value, prelude)
            return replaced(expr, target=target, value=value)
        lowered = self.hoist_value_side_effects(expr, prelude)
        if not any(isinstance(node, FunctionCallNode) for node in lowered.walk()):
            # A pure read has no effect, and WGSL rejects it as a statement.
            return None
        return lowered

    def lower_if_statement(self, stmt):
        prelude = []
        condition = stmt.condition
        if contains_value_side_effect(condition):
            condition = self.hoist_value_side_effects(condition, prelude)
        else_branch = self.lower_else_if_chain(stmt.else_branch)
        if condition is stmt.condition and else_branch is stmt.else_branch:
            return None
        return [*prelude, replaced(stmt, condition=condition, else_branch=else_branch)]

    def lower_else_if_chain(self, else_branch):
        """Move an else-if whose condition has side effects into its own block.

        The nested ``if`` is then generated as a statement, so its side effects
        run only when the earlier conditions are false.
        """
        if not isinstance(else_branch, IfNode):
            return else_branch
        if contains_value_side_effect(else_branch.condition):
            return BlockNode([else_branch])
        nested = self.lower_else_if_chain(else_branch.else_branch)
        if nested is else_branch.else_branch:
            return else_branch
        return replaced(else_branch, else_branch=nested)

    def loop_guard(self, condition):
        """The statements that test a loop condition with side effects and break."""
        prelude = []
        condition = self.hoist_value_side_effects(condition, prelude)
        exit_test = UnaryOpNode("!", condition)
        return [*prelude, IfNode(exit_test, BlockNode([BreakNode()]))]

    def lower_while_statement(self, stmt):
        if not contains_value_side_effect(stmt.condition):
            return None
        body = BlockNode([*self.loop_guard(stmt.condition), stmt.body])
        return [LoopNode(body)]

    def lower_for_statement(self, stmt):
        if statement_operand_side_effects(stmt.update):
            raise ValueError(
                "WGSL target cannot lower a for-loop update that uses an increment, "
                "decrement or assignment as a value; WGSL has no such expressions "
                "and the update runs after every continue"
            )
        prelude, init = self.lower_for_initializer(stmt.init)
        condition, body = stmt.condition, stmt.body
        if contains_value_side_effect(condition):
            body = BlockNode([*self.loop_guard(condition), body])
            condition = None
        if init is stmt.init and body is stmt.body:
            return None
        loop = replaced(stmt, init=init, condition=condition, body=body)
        if not prelude:
            return [loop]
        return [BlockNode([*prelude, loop])]

    def lower_for_initializer(self, init):
        prelude = []
        if isinstance(init, VariableNode):
            if not contains_value_side_effect(init.initial_value):
                return prelude, init
            value = self.hoist_value_side_effects(init.initial_value, prelude)
            return prelude, replaced(init, initial_value=value)
        if isinstance(init, ExpressionStatementNode):
            if not statement_operand_side_effects(init.expression):
                return prelude, init
            expression = self.lower_statement_expression(init.expression, prelude)
            if expression is None:
                return prelude, None
            return prelude, replaced(init, expression=expression)
        if init is not None and statement_operand_side_effects(init):
            return prelude, self.lower_statement_expression(init, prelude)
        return prelude, init

    def hoist_value_side_effects(self, expr, prelude, conditional=False):
        """``expr`` with its side effects appended to ``prelude`` as statements.

        ``conditional`` marks a position the source may skip, where hoisting
        would run the side effect unconditionally.
        """
        if expr is None or not contains_value_side_effect(expr):
            return expr
        if is_value_side_effect(expr) and conditional:
            raise ValueError(
                "WGSL target cannot lower an increment, decrement or assignment used "
                "as a value in the right operand of && or ||, or in a ternary branch; "
                "WGSL has no such expressions and hoisting it would run it "
                "unconditionally"
            )
        if is_increment(expr):
            return self.hoist_increment(expr, prelude)
        if isinstance(expr, AssignmentNode):
            return self.hoist_assignment(expr, prelude)
        hoist = self.hoist_value_side_effects
        if isinstance(expr, BinaryOpNode):
            left = hoist(expr.left, prelude, conditional)
            right_conditional = conditional or expr.operator in SHORT_CIRCUIT_OPERATORS
            right = hoist(expr.right, prelude, right_conditional)
            return replaced(expr, left=left, right=right)
        if isinstance(expr, TernaryOpNode):
            return replaced(
                expr,
                condition=hoist(expr.condition, prelude, conditional),
                true_expr=hoist(expr.true_expr, prelude, True),
                false_expr=hoist(expr.false_expr, prelude, True),
            )
        if isinstance(expr, UnaryOpNode):
            return replaced(expr, operand=hoist(expr.operand, prelude, conditional))
        if isinstance(expr, (FunctionCallNode, ConstructorNode)):
            if contains_value_side_effect(
                getattr(expr, "function", None)
            ) or contains_value_side_effect(getattr(expr, "named_arguments", None)):
                self.reject_unordered_side_effect(expr)
            arguments = [
                hoist(argument, prelude, conditional) for argument in expr.arguments
            ]
            return replaced(expr, arguments=arguments)
        if isinstance(expr, ArrayLiteralNode):
            elements = [hoist(item, prelude, conditional) for item in expr.elements]
            return replaced(expr, elements=elements)
        if isinstance(expr, MemberAccessNode):
            return replaced(
                expr, object_expr=hoist(expr.object_expr, prelude, conditional)
            )
        if isinstance(expr, SwizzleNode):
            return replaced(
                expr, vector_expr=hoist(expr.vector_expr, prelude, conditional)
            )
        if isinstance(expr, ArrayAccessNode):
            array_expr = hoist(expr.array_expr, prelude, conditional)
            index_expr = hoist(expr.index_expr, prelude, conditional)
            return replaced(expr, array_expr=array_expr, index_expr=index_expr)
        if isinstance(expr, CastNode):
            return replaced(
                expr, expression=hoist(expr.expression, prelude, conditional)
            )
        self.reject_unordered_side_effect(expr)

    def reject_unordered_side_effect(self, expr):
        raise ValueError(
            "WGSL target cannot order an increment, decrement or assignment used as "
            f"a value inside {describe(expr)}"
        )

    def repeatable_lvalue(self, lvalue, context):
        """Check that ``lvalue`` may be evaluated twice without observable effect."""
        for node in lvalue.walk():
            if isinstance(node, REPEATABLE_NODE_TYPES) and not is_value_side_effect(
                node
            ):
                continue
            raise ValueError(
                f"WGSL target cannot lower {context} used as a value: its l-value "
                f"contains {describe(node)}, which the lowering would evaluate twice"
            )
        return lvalue

    def hoist_increment(self, expr, prelude):
        operand = self.hoist_value_side_effects(expr.operand, prelude)
        self.repeatable_lvalue(operand, f"'{expr.operator}'")
        update = ExpressionStatementNode(UnaryOpNode(expr.operator, operand))
        if not expr.is_postfix:
            prelude.append(update)
            return operand
        prior_name = self.next_prior_value_name()
        prelude.append(VariableNode(prior_name, None, operand, is_mutable=False))
        prelude.append(update)
        return IdentifierNode(prior_name)

    def hoist_assignment(self, expr, prelude):
        target = self.hoist_value_side_effects(expr.target, prelude)
        value = self.hoist_value_side_effects(expr.value, prelude)
        self.repeatable_lvalue(target, f"assignment '{expr.operator}'")
        prelude.append(
            ExpressionStatementNode(replaced(expr, target=target, value=value))
        )
        return target

    def next_prior_value_name(self):
        index = getattr(self, "_prior_value_count", 0)
        self._prior_value_count = index + 1
        return f"{PRIOR_VALUE_PREFIX}{index}"
